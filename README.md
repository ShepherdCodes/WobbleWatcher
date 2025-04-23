# WobbleWatcher
Authors: Prateek Boga, Arno Riffeser, 2025.
Python code to calculate seeing and seeing variability from the guiding camera images at Wendelstein Space Telescope

Wendelstein Space telescope (WST) sits on top of Mount Wendelstein in the Bavarian Alps at ~1800 m. The seeing monitor when functional points at the polar star, which might have some discrepancies if the observation is done on a different airmass (direction). This python code can be applied to the guiding camera fits files to analyse each fits file, find the "background stars" and calculate the seeing using the Full Width of Half Maximum (FWHM) of the Point Spread Function (PSF) of multiple stars. It creates an animation of the night, the seeing throughout the night, and it's variability. If the weather conditions are bad, a "warning" of no detections is given. 

Note: For accurate calculation of seeing, please vary the fwhm and threshold in the following line of the code and see what works best by cross checking the animation. 
daofind = DAOStarFinder(fwhm=3.0, threshold=10 * std)
