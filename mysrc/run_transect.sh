#!/usr/bin/env bash
set -euo pipefail

source ~/miniforge3/bin/activate orthoriry

flightname=""
flight_date=""
transec_prefix=""
transec_number=""
name_imu=""
file_imu=""
flightnote=""
calib_transect=""
calib_file_input=""
overwriteflag=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --flightname)
      [[ $# -ge 2 ]] || { echo "Error: --flightname requires a value." >&2; exit 2; }
      flightname=$2; shift 2 ;;
    --flight-date)
      [[ $# -ge 2 ]] || { echo "Error: --flight-date requires a value." >&2; exit 2; }
      flight_date=$2; shift 2 ;;
    --transect-prefix)
      [[ $# -ge 2 ]] || { echo "Error: --transect-prefix requires a value." >&2; exit 2; }
      transec_prefix=$2; shift 2 ;;
    --transect-number)
      [[ $# -ge 2 ]] || { echo "Error: --transect-number requires a value." >&2; exit 2; }
      transec_number=$2; shift 2 ;;
    --imu-name)
      [[ $# -ge 2 ]] || { echo "Error: --imu-name requires a value." >&2; exit 2; }
      name_imu=$2; shift 2 ;;
    --imu-file)
      [[ $# -ge 2 ]] || { echo "Error: --imu-file requires a value." >&2; exit 2; }
      file_imu=$2; shift 2 ;;
    --note)
      [[ $# -ge 2 ]] || { echo "Error: --note requires a value." >&2; exit 2; }
      flightnote=$2; shift 2 ;;
    --calib-transect)
      [[ $# -ge 2 ]] || { echo "Error: --calib-transect requires a value." >&2; exit 2; }
      calib_transect=$2; shift 2 ;;
    --calib-file)
      [[ $# -ge 2 ]] || { echo "Error: --calib-file requires a value." >&2; exit 2; }
      calib_file_input=$2; shift 2 ;;
    --overwrite)
      [[ $# -ge 2 ]] || { echo "Error: --overwrite requires true or false." >&2; exit 2; }
      overwriteflag=$2; shift 2 ;;
    *)
      echo "Error: unknown option: $1" >&2
      exit 2
      ;;
  esac
done

for required in flightname flight_date transec_prefix transec_number name_imu file_imu flightnote; do
  if [[ -z "${!required}" ]]; then
    echo "Error: --${required//_/-} is required." >&2
    exit 2
  fi
done
if [[ -z "$overwriteflag" ]]; then
  echo "Error: --overwrite true|false is required." >&2
  exit 2
fi

base_name=$(printf "%s%05d" "$transec_prefix" "$transec_number")

run_calibration=false
if [[ -n "$calib_transect" && "$calib_transect" -eq "$transec_number" ]]; then
  run_calibration=true
elif [[ -z "$calib_file_input" ]]; then
  echo "Error: --calib-file must be an absolute path when --calib-transect is omitted or different." >&2
  exit 2
else
  if [[ "$calib_file_input" != /* ]]; then
    echo "Error: --calib-file must be a full absolute path: $calib_file_input" >&2
    exit 2
  fi
  if [[ ! -f "$calib_file_input" ]]; then
    echo "Error: calibration file not found: $calib_file_input" >&2
    exit 2
  fi
  if [[ "$calib_file_input" =~ calib0*([0-9]+) ]]; then
    path_calib_transect=$((10#${BASH_REMATCH[1]}))
  else
    echo "Error: could not extract calib number from calibration path: $calib_file_input" >&2
    exit 2
  fi
  if [[ -n "$calib_transect" && "$calib_transect" -ne "$path_calib_transect" ]]; then
    echo "Error: --calib-transect $calib_transect does not match calibration path number $path_calib_transect." >&2
    exit 2
  fi
  calib_transect=$path_calib_transect
fi

calib_base_name=$(printf "%s%05d" "$transec_prefix" "$calib_transect")
calib_name=$(printf "calib%05d" "$calib_transect")
calib_transect_name="${calib_base_name}-${name_imu}-${calib_name}"
calib_config_file="config/calibration"-"$flightname"-"$base_name"-"$name_imu".yaml

if [[ "$run_calibration" == true ]]; then
  # Calibrating this transect: use the calibration YAML and its output.
  calib_file="/data/shared/PIPER/$flightname/Transects/$calib_transect_name/calib/imu_camera_calibration_${name_imu}.json"
else
  # Reusing another transect's calibration: use the validated full path.
  calib_file="$calib_file_input"
fi

overwrite=false
case "${overwriteflag,,}" in
  true|1|yes) overwrite=true ;;
esac

bas_name="${base_name}-${name_imu}-${calib_name}"
name_capital=$(echo "$name_imu" | cut -d'_' -f1 | tr '[:lower:]' '[:upper:]')
root=/data/shared/PIPER/$flightname/Transects/$bas_name
config_file=config/config-$flightname-$bas_name.yaml
ortho_dir=$root/full_ortho_f1_$name_imu
corr_dir=$root/${name_imu}_correlation_timeseries
webm=$ortho_dir/full_ortho_f1_track_${name_imu}.webm
log_dir=$root/run_logs
mkdir -p "$log_dir"

echo '######################################'
printf '# %s %s %s %s #\n' "$flightname" "$bas_name" "$name_imu" "$calib_name"
echo '######################################'

task_complete() {
  local number=$1
  local sentinel=$2
  case "$number" in
    1) find "$sentinel" -maxdepth 1 -type f -name "*.tif" -print -quit 2>/dev/null | grep -q . ;;
    4) find "$sentinel" -maxdepth 1 -type f -name "*_ORTHO.tif" -print -quit 2>/dev/null | grep -q . ;;
    *) [[ -e "$sentinel" ]] ;;
  esac
}

run_task() {
  local number=$1
  local description=$2
  local sentinel=$3
  shift 3
  if [[ "$overwrite" == false ]] && task_complete "$number" "$sentinel"; then
    echo "[$number] Skipping $description; already completed: $sentinel"
    return 0
  fi
  local log_name
  log_name=$(printf "%s" "$description" | tr " " "_" | tr -cd "[:alnum:]_-" | tr "[:upper:]" "[:lower:]")
  local log_file="$log_dir/${number}_${log_name}.log"
  echo "[$number] Running $description"
  if "$@" >"$log_file" 2>&1; then
    echo "[$number] Completed $description"
  else
    status=$?
    echo "[$number] FAILED $description (exit $status); see $log_file"
    return "$status"
  fi
}

# 0. Initialize
# 0.0transect structure and link raw images
run_task 0 "transect initialization" "$root/io/camera.yaml" \
  python init_transect.py \
    --transects-dir /data/shared/PIPER/$flightname/Transects \
    --prefix "$transec_prefix" --number "$transec_number" --output-name "$bas_name" \
    --manifest /data/shared/PIPER/$flightname/Transects/report_legs.csv \
    --raw-dir /data/shared/PIPER/$flightname/bas

log_dir=$root/run_logs
mkdir -p "$log_dir"

# 0.1 Estimate image/IMU time lag from optical flow
motion_dir="$root/image_sequence_motion_$name_imu"
lag_file="$motion_dir/best_time_lag.txt"
run_task 0.1 "image motion and time-lag estimation" "$lag_file" \
      python image_sequence_motion.py \
    --images-dir "$root/tif_f1" \
    --imu "$file_imu" \
    --output-dir "$motion_dir"

timelag=$(awk -F= '$1 == "time_lag_seconds" {print $2}' "$lag_file")
if [[ -z "$timelag" ]]; then
  echo "[0.1] FAILED: no time_lag_seconds found in $lag_file"
  exit 1
fi
echo "[0.1] Using estimated time lag: ${timelag} s"

# 1. Crop input images to 320x241
if [[ "$overwrite" == true ]]; then
  crop_args=(python crop_tif_f1_320.py "$root/tif_f1" --output-dir "$root/tif_f1_320" --overwrite)
else
  crop_args=(python crop_tif_f1_320.py "$root/tif_f1" --output-dir "$root/tif_f1_320")
fi
run_task 1 "320x241 crop" "$root/tif_f1_320" "${crop_args[@]}"

# 2. Create transect DEM
run_task 2 "transect DEM" "$root/dem/${base_name}_rgealti.tif" \
  python create_transect_dem.py "$root" \
    --image-dir "$root/tif_f1" --imu-dir /data/shared/PIPER/$flightname/imu \
    --dem-root /data/shared/RGEALTI_IGN --buffer-km 5

# 3. Generate orthorectification configuration
# 3.0 Calibrate boresight only when the selected calibration transect is this
#     transect. Otherwise, retain the explicitly supplied calibration file.
if [[ "$run_calibration" == true ]]; then
  run_task 3.0 "IMU-camera calibration" "$calib_file" \
    python calibrate_orthority_imu.py "$calib_config_file"
else
  echo "[3.0] Skipping IMU-camera calibration; using supplied calibration: $calib_file"
fi

# 3.1 config file
run_task 3 "configuration generation" "$config_file" \
  python generate_config.py \
    --flightname "$flightname" --transectname "$bas_name" --imuname "$name_imu" \
    --imufilename "$file_imu" --flightdate "$flight_date" \
    --defaultcalibration "$calib_file" --calib-transect "$calib_transect" \
    --note "$flightnote" --time-shift "$timelag"

# 4. Run orthorectification
run_task 4 "orthorectification" "$ortho_dir" \
  python runOrtho.py --config "$config_file"

# 5. Compute performance:
# 5.1 ortho correlation
corr_csv="$corr_dir/ortho_correlation_timeseries_${name_capital}.csv"
run_task 5 "ortho correlation" "$corr_csv" \
  python correlate_ortho_sequence.py "$ortho_dir" \
    --imu "$file_imu"  \
    --output-dir "$corr_dir"

# 5.2 Estimate time-varying IMU drift
drift_dir="$root/imu_drift_$name_imu"
drift_csv="$drift_dir/imu_drift_timeseries.csv"
run_task 5.2 "IMU drift estimation" "$drift_csv" \
  python estimate_imu_drift.py --config "$config_file"

# 6. Generate WebM
if [[ "$overwrite" == true ]]; then
  webm_args=(python geotiff_dir_to_webm.py "$ortho_dir" "$config_file" --output "$webm" --overwrite)
else
  webm_args=(python geotiff_dir_to_webm.py "$ortho_dir" "$config_file" --output "$webm")
fi
run_task 6 "WebM generation" "$webm" "${webm_args[@]}"

# 7. Publish results to website
webdir=/data/shared/PIPER/website/data/${flightname}-${bas_name}
website_sentinel="$webdir/$(basename "$webm")"
if [[ "$overwrite" == false && -e "$website_sentinel" ]]; then
  echo "[7] Skipping website publication; already completed: $website_sentinel"
else
  website_log="$log_dir/7_website_publication.log"
  echo "[7] Publishing website results"
  if {
    mkdir -p "$webdir"
    cp "$webm" "$webdir"
    cp "$ortho_dir/full_ortho_f1_track_${name_imu}.manifest.json" "$webdir"
    cp "$corr_csv" "$webdir"
    cp "$drift_csv" "$webdir"
    cp "$config_file" "$webdir"
  } >"$website_log" 2>&1; then
    echo "[7] Completed website publication"
  else
    status=$?
    echo "[7] FAILED website publication (exit $status); see $website_log"
    exit "$status"
  fi
fi
