import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.background import MMMBackground
from astropy.stats import sigma_clipped_stats
import glob
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle

# Constants
PIXEL_SCALE = 0.165  # arcseconds per pixel for WST
MASK_RADIUS = 76  # Pixels, change as needed
MAIN_OBJECT_COORDS = (230, 100)  # Change based on image (x, y)
fits_files = sorted(glob.glob("*.fits"))  # Load all FITS files

# Lists to store seeing values
timestamps = []
averaged_seeing_values = []  # Store final averaged seeing values per timestamp

class FITS_Animation:
    def __init__(self, files, interval=100):
        self.files = files
        self.interval = interval
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ima = None
        self.colorbar = None
        self.anim = None

    def process_fits(self, filename):
        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            timestamp = header.get('DATE-OBS', filename)
            timestamps.append(timestamp)

            # Mask the main object
            y, x = np.indices(data.shape)
            mask = (np.sqrt((x - MAIN_OBJECT_COORDS[0])**2 + (y - MAIN_OBJECT_COORDS[1])**2) < MASK_RADIUS)
            masked_data = np.copy(data).astype(float)
            masked_data[mask] = np.nanmedian(data)

            # Find background statistics
            mean, median, std = sigma_clipped_stats(masked_data, sigma=3.0)
            bkg = MMMBackground()(masked_data)
            masked_data -= median

            # **Fixed FWHM & Threshold for Star Detection**
            daofind = DAOStarFinder(fwhm=3.0, threshold=10 * std)  # Play with these along side visual inspection to get best outcomes
            sources = daofind(masked_data)

            detected_sources = sources if sources is not None and len(sources) > 0 else []

            # **Seeing Calculation with Fixed FWHM = 10**
            all_seeing_values = []
            for src in detected_sources:
                seeing = self.calculate_seeing(masked_data, int(src['xcentroid']), int(src['ycentroid']), fwhm=10)
                if not np.isnan(seeing):
                    all_seeing_values.append(seeing)

            # Average seeing values
            avg_seeing = np.nanmedian(all_seeing_values) if all_seeing_values else np.nan
            averaged_seeing_values.append(avg_seeing)

            return data, masked_data, detected_sources, bkg



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
        
        if len(x_indices) > 100 or len(y_indices) > 100:  # Reasonable upper limit
            print(f"Rejecting FWHM: {fwhm_pixels} (too many pixels at half-max: {len(x_indices)})")
            return np.nan
        
        if fwhm_pixels < 2.0:  # Discard detections that are too sharp
            return np.nan

        
        return fwhm_pixels * PIXEL_SCALE

    def update(self, nr):
        if nr < len(self.files):
            filename = self.files[nr]
            data, masked_data, detected_sources, bkg = self.process_fits(filename)

            self.ax.clear()
            self.ax.set_title(f"Masked FITS Image: {filename}")
            self.ima = self.ax.imshow(masked_data, origin='lower', cmap='viridis',
                                      vmin=-5 * np.std(masked_data), vmax=5 * np.std(masked_data))

            # Add hatched circle for masked region
            mask_circle = Circle(MAIN_OBJECT_COORDS, MASK_RADIUS, edgecolor='red',
                                 facecolor='none', hatch='///', linewidth=2, label="Masked Region")
            self.ax.add_patch(mask_circle)

            if detected_sources:
                for src in detected_sources:
                    x, y = src['xcentroid'], src['ycentroid']
                    flux = src['flux']
                    snr = flux / bkg
                    
                    # Hollow circle
                    star_circle = Circle((x, y), radius=5, edgecolor='blue', facecolor='none', linewidth=1.5)
                    self.ax.add_patch(star_circle)

                    # Add S/N label
                    self.ax.text(x + 5, y + 5, f"S/N: {snr:.1f}", color='white', fontsize=6)

            self.ax.legend(loc='upper right')
            
            # Add a scale bar for 1 arcsecond (6 pixels, based on PIXEL_SCALE)
            scale_length_pixels = int(1 / PIXEL_SCALE)  # 6 pixels for 1 arcsecond
            self.ax.plot(
                [10, 10 + scale_length_pixels], [10, 10],  # Scale bar starting at (10, 10)
                color='white', linewidth=3
            )
            self.ax.text(10 + scale_length_pixels + 2, 10, "1\"",
                         color='white', fontsize=8, verticalalignment='bottom')

            if self.colorbar is None:
                self.colorbar = self.fig.colorbar(self.ima, ax=self.ax, orientation='vertical')
                self.colorbar.set_label('Pixel Intensity')
            else:
                self.colorbar.update_normal(self.ima)

        return self.ima,

    def run(self):
        self.anim = FuncAnimation(self.fig, self.update, interval=self.interval, frames=len(self.files), blit=False)
        writer = FFMpegWriter(fps=5, metadata={'title': 'FITS Masked Animation'})
        self.anim.save('fits_masked_animation.mp4', writer=writer, dpi=100)
        print("Animation saved as 'fits_masked_animation.mp4'")
        self.plot_seeing_vs_time()
    
    def plot_seeing_vs_time(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Convert timestamps to numeric values for proper plotting
        numeric_timestamps = np.arange(len(timestamps))
        
        # Filter seeing values below 10.0
        filtered_seeing = [s for s in averaged_seeing_values if s < 10.0]
        
        # Plotting the seeing over time
        ax1.scatter(numeric_timestamps, averaged_seeing_values, marker='o', color='b', label="Seeing Data")
        ax1.set_xlabel("Observation Time")
        ax1.set_ylabel("Seeing (arcseconds)")
        ax1.set_title("Seeing Variation Over Time")
        
        # Calculate mean and standard deviation for the filtered seeing values
        if filtered_seeing:  # Ensure there's at least one value below threshold
            avg_seeing = np.mean(filtered_seeing)
            ax1.axhline(avg_seeing, color='r', linestyle='--', label=f'Mean Seeing = {avg_seeing:.2f}')
            
            mean_seeing = np.mean(filtered_seeing)
            std_seeing = np.std(filtered_seeing)

            # Define 1-sigma and 2-sigma boundaries
            lower_1sigma = mean_seeing - std_seeing
            upper_1sigma = mean_seeing + std_seeing
            lower_2sigma = mean_seeing - 2 * std_seeing
            upper_2sigma = mean_seeing + 2 * std_seeing
            
            # Fill the 1-sigma and 2-sigma regions
            ax1.fill_between(numeric_timestamps, lower_1sigma, upper_1sigma, color='orange', alpha=0.6, label=r'1-$\sigma$ Region')
            ax1.fill_between(numeric_timestamps, lower_2sigma, upper_2sigma, color='orange', alpha=0.3, label=r'2-$\sigma$ Region')

        # Dynamically select fewer timestamps for x-axis labels
        num_ticks = min(8, len(timestamps))  # Limit to 8 ticks max
        tick_indices = np.linspace(0, len(timestamps) - 1, num_ticks, dtype=int)
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels([timestamps[i] for i in tick_indices], rotation=30, ha="right")
        ax1.grid()
        ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
        # Plotting the variance of seeing over time
        ax2.set_xlabel("Observation Time")
        ax2.set_ylabel("Variance of Seeing (arcseconds)")
        ax2.set_title("Seeing Variability Over Time")
        
        # Calculate and plot variance only if filtered seeing is not empty
        if filtered_seeing:
            rolling_variance = np.convolve(filtered_seeing, np.ones(5)/5, mode='valid')  # Rolling variance with window size of 5
            ax2.plot(numeric_timestamps[2:len(rolling_variance) + 2], rolling_variance, color='g', label="Rolling Variance")
        else:
            # If filtered_seeing is empty, plot an empty variance line
            ax2.plot([], [], color='g', label="Rolling Variance (No Data)")
        
        ax2.grid()
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
        plt.tight_layout()  # Ensure everything fits inside the figure
        plt.savefig("seeing_vs_time_variability.png")
        plt.close()
        
        print("Seeing and variability plot saved as 'seeing_vs_time_variability.png'")

if fits_files:
    animation = FITS_Animation(fits_files)
    animation.run()
else:
    print("No FITS files found.")
