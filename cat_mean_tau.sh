
#!/bin/bash
# Cat files 
set file_f='/glade/campaign/cgd/oce/people/bachman/ETP_1_20_tides/HOURLY/ocean_hourly__1993_001*.nc'
for year in 1993; do
	for day in {001..366}; do
		if (($day <= 366 && $year ==1993)); then
			echo Processing files "$year"_"$day" 
			ncra -O -v taux,tauy $file_f /glade/campaign/cgd/oce/people/bachman/ETP_1_20_tides/HOURLY/ocean_hourly__"$year"_"$day"*.nc /glade/scratch/pmora/mean_tauxy.nc
			file_f='/glade/scratch/pmora/mean_tauxy.nc'
			fi
	done
done

