import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.background import MMMBackground
from astropy.stats import sigma_clipped_stats
import glob
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Constants
PIXEL_SCALE = 0.165  # arcseconds per pixel for WST
REGION_SIZE = 120  # Square region size in pixels
MAIN_OBJECT_COORDS = (230, 100)  # Center of region (x, y)
fits_files = sorted(glob.glob("*.fits"))  # Load all FITS files

# Lists to store seeing values
timestamps = []
averaged_seeing_values = []  # Store final averaged seeing values per timestamp

class FITS_Animation:
    def __init__(self, files, interval=100):
        self.files = files
        self.interval = interval
        self.fig = plt.figure(figsize=(12, 8))
        
        # Create grid layout
        gs = self.fig.add_gridspec(4, 4)
        self.ax_main = self.fig.add_subplot(gs[:3, :3])
        self.ax_histx = self.fig.add_subplot(gs[3, :3])
        self.ax_histy = self.fig.add_subplot(gs[:3, 3])
        
        self.ima = None
        self.colorbar = None
        self.anim = None
        self.region_patch = None

    def process_fits(self, filename):
        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            timestamp = header.get('DATE-OBS', filename)
            timestamps.append(timestamp)

            # Define square region boundaries
            x_center, y_center = MAIN_OBJECT_COORDS
            half_size = REGION_SIZE // 2
            x_min = max(x_center - half_size, 0)
            x_max = min(x_center + half_size, data.shape[1])
            y_min = max(y_center - half_size, 0)
            y_max = min(y_center + half_size, data.shape[0])

            # Create rectangular mask for the main object region
            mask = np.zeros_like(data, dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            masked_data = np.copy(data).astype(float)
            masked_data[mask] = np.nanmedian(data)  # Replace masked region with median

            # Find background statistics from masked data
            mean, median, std = sigma_clipped_stats(masked_data, sigma=3.0)
            bkg = MMMBackground()(masked_data)
            masked_data -= median

            # Star detection on MASKED data (excluding the main object region)
            daofind = DAOStarFinder(fwhm=3.0, threshold=10 * std)
            sources = daofind(masked_data)
            detected_sources = sources if sources is not None and len(sources) > 0 else []

            # Seeing calculation
            all_seeing_values = []
            for src in detected_sources:
                seeing = self.calculate_seeing(masked_data, int(src['xcentroid']), int(src['ycentroid']), fwhm=10)
                if not np.isnan(seeing):
                    all_seeing_values.append(seeing)

            avg_seeing = np.nanmedian(all_seeing_values) if all_seeing_values else np.nan
            averaged_seeing_values.append(avg_seeing)

            return data, masked_data, (x_min, x_max, y_min, y_max), detected_sources, bkg, median, std

    def calculate_seeing(self, data, x_star, y_star, fwhm):
        if x_star is None or y_star is None:
            return np.nan

        cutout_size = 30
        half_size = cutout_size // 2
        y_min = max(y_star - half_size, 0)
        y_max = min(y_star + half_size + 1, data.shape[0])
        x_min = max(x_star - half_size, 0)
        x_max = min(x_star + half_size + 1, data.shape[1])

        star_cutout = data[y_min:y_max, x_min:x_max]
        if star_cutout.size == 0:
            return np.nan

        max_value = np.max(star_cutout)
        half_max = max_value * 0.5
        half_max_mask = star_cutout >= half_max
        y_indices, x_indices = np.where(half_max_mask)

        if len(x_indices) > 0:
            fwhm_x = np.max(x_indices) - np.min(x_indices) + 1
            fwhm_y = np.max(y_indices) - np.min(y_indices) + 1
        else:
            fwhm_x = fwhm_y = 0

        fwhm_pixels = (fwhm_x + fwhm_y) / 2.0
        
        if len(x_indices) > 100 or len(y_indices) > 100:
            return np.nan
        
        if fwhm_pixels < 2.0:
            return np.nan

        return fwhm_pixels * PIXEL_SCALE

    def update(self, nr):
        if nr < len(self.files):
            filename = self.files[nr]
            data, masked_data, region_coords, detected_sources, bkg, median, std = self.process_fits(filename)
            x_min, x_max, y_min, y_max = region_coords

            # Clear axes
            self.ax_main.clear()
            self.ax_histx.clear()
            self.ax_histy.clear()

            # Main image (show original data but with masked region)
            self.ax_main.set_title(f"FITS Image: {filename}")
            self.ima = self.ax_main.imshow(data, origin='lower', cmap='viridis',
                                        vmin=median-5*std, vmax=median+5*std)

            # Add rectangle for masked region
            if self.region_patch:
                self.region_patch.remove()
            self.region_patch = Rectangle((x_min, y_min), REGION_SIZE, REGION_SIZE,
                                      edgecolor='red', facecolor='none', linewidth=1.5,
                                      linestyle='--', label="Masked Region")
            self.ax_main.add_patch(self.region_patch)

            # Add detected stars with S/N labels
            if detected_sources:
                for src in detected_sources:
                    x, y = src['xcentroid'], src['ycentroid']
                    flux = src['flux']
                    snr = flux / bkg
                    
                    # Hollow circle for star
                    star_circle = Circle((x, y), radius=5, edgecolor='blue',
                                      facecolor='none', linewidth=1.5)
                    self.ax_main.add_patch(star_circle)
                    
                    # S/N label
                    self.ax_main.text(x + 8, y + 8, f"S/N: {snr:.1f}",
                                   color='white', fontsize=8)

            # Add scale bar
            scale_length_pixels = int(1 / PIXEL_SCALE)
            self.ax_main.plot([10, 10 + scale_length_pixels], [10, 10],
                           color='white', linewidth=3)
            self.ax_main.text(10 + scale_length_pixels + 2, 10, "1\"",
                            color='white', fontsize=8, verticalalignment='bottom')

            # Create properly aligned histograms for the region
            region_data = data[y_min:y_max, x_min:x_max]
            self.create_aligned_histograms(region_data, x_min, y_min)

            # Update colorbar if it exists, otherwise create it
            if self.colorbar is None:
                divider = make_axes_locatable(self.ax_main)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.colorbar = plt.colorbar(self.ima, cax=cax)
                self.colorbar.set_label('Pixel Intensity')
            else:
                self.colorbar.update_normal(self.ima)

            self.ax_main.legend(loc='upper right')

        return self.ima,

    def create_aligned_histograms(self, region_data, x_min, y_min):
        """Create histograms properly aligned with the region using 50 bins."""
        # Number of bins
        bins = 20
        
        # Horizontal histogram (x-axis profile)
        x_profile = np.sum(region_data, axis=0)
        hist_x, bin_edges_x = np.histogram(np.arange(x_min, x_min + len(x_profile)),
                              weights=x_profile, bins=bins)
        bin_centers_x = (bin_edges_x[:-1] + bin_edges_x[1:]) / 2
        self.ax_histx.step(bin_centers_x, hist_x, where='mid', color='blue')
        self.ax_histx.set_xlim(self.ax_main.get_xlim())
        self.ax_histx.set_ylabel('Sum Intensity')
        self.ax_histx.grid(True)
        
        # Vertical histogram (y-axis profile)
        y_profile = np.sum(region_data, axis=1)
        hist_y, bin_edges_y = np.histogram(np.arange(y_min, y_min + len(y_profile)),
                              weights=y_profile, bins=bins)
        bin_centers_y = (bin_edges_y[:-1] + bin_edges_y[1:]) / 2
        self.ax_histy.step(hist_y, bin_centers_y, where='mid', color='blue')
        self.ax_histy.set_ylim(self.ax_main.get_ylim())
        self.ax_histy.set_xlabel('Sum Intensity')
        self.ax_histy.grid(True)
        
        # Adjust layout
        self.ax_histx.set_xticks([])
        self.ax_histy.set_yticks([])

    def run(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=self.interval,
                               frames=len(self.files), blit=False)
        writer = FFMpegWriter(fps=5, metadata={'title': 'FITS Analysis Animation'})
        self.anim.save('fits_analysis_animation.mp4', writer=writer, dpi=100)
        print("Animation saved as 'fits_analysis_animation.mp4'")
        self.plot_seeing_vs_time()

    def plot_seeing_vs_time(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        numeric_timestamps = np.arange(len(timestamps))
        filtered_seeing = [s for s in averaged_seeing_values if s < 10.0]
        
        ax1.scatter(numeric_timestamps, averaged_seeing_values, marker='o', color='b', label="Seeing Data")
        ax1.set_xlabel("Observation Time")
        ax1.set_ylabel("Seeing (arcseconds)")
        ax1.set_title("Seeing Variation Over Time")
        
        if filtered_seeing:
            avg_seeing = np.mean(filtered_seeing)
            ax1.axhline(avg_seeing, color='r', linestyle='--', label=f'Mean Seeing = {avg_seeing:.2f}')
            
            mean_seeing = np.mean(filtered_seeing)
            std_seeing = np.std(filtered_seeing)
            lower_1sigma = mean_seeing - std_seeing
            upper_1sigma = mean_seeing + std_seeing
            lower_2sigma = mean_seeing - 2 * std_seeing
            upper_2sigma = mean_seeing + 2 * std_seeing
            
            ax1.fill_between(numeric_timestamps, lower_1sigma, upper_1sigma, color='orange', alpha=0.6, label=r'1-$\sigma$ Region')
            ax1.fill_between(numeric_timestamps, lower_2sigma, upper_2sigma, color='orange', alpha=0.3, label=r'2-$\sigma$ Region')

        num_ticks = min(8, len(timestamps))
        tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels([timestamps[i] for i in tick_indices], rotation=30, ha="right")
        ax1.grid()
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
        ax2.set_xlabel("Observation Time")
        ax2.set_ylabel("Variance of Seeing (arcseconds)")
        ax2.set_title("Seeing Variability Over Time")
        
        if filtered_seeing:
            rolling_variance = np.convolve(filtered_seeing, np.ones(5)/5, mode='valid')
            ax2.plot(numeric_timestamps[2:len(rolling_variance) + 2], rolling_variance, color='g', label="Rolling Variance")
        else:
            ax2.plot([], [], color='g', label="Rolling Variance (No Data)")
        
        ax2.grid()
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig("seeing_vs_time_variability.png")
        plt.close()
        
        print("Seeing and variability plot saved as 'seeing_vs_time_variability.png'")

if fits_files:
    animation = FITS_Animation(fits_files)
    animation.run()
else:
    print("No FITS files found.")
