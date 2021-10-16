mv *txt /home/bill/

# /home/videos/2021-01-29_15-22-54
#IN_DATA_DIR="/home/videos/2021-02-02_15-50-21"
#OUT_DATA_DIR="/home/videos/2021-02-02_15-50-21/frames"
IN_DATA_DIR="/home/bill/datasets/4k/2021-07-16_16-49-34"
OUT_DATA_DIR="/home/bill/datasets/4k/2021-07-16_16-49-34/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  ffmpeg -i "${video}" -r 20 -q:v 1 "${out_name}"
done

mv /home/bill/*txt .