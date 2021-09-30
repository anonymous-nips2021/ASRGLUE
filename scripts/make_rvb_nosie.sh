#!/usr/bin/env bash
#source /opt/share/etc/gcc-5.4.0.sh
set -e -o pipefail

stage=0
nj=25
SNR="10" # 在这个位置控制添加噪声的强度，SNR 是信噪比，越小噪声越大。 SNR 通常取 -5 0 5 10
#norvb_datadir=data/librispeech/train_960_cleaned

#rvb_affix=_rvb_noise_$SNR
SP=1
data=
num_data_reps=1
sample_rate=16000
max_jobs_run=20
cmd="run.pl"
#SNR="5"
#. ./cmd.sh
#. ./path.sh
. ./utils/parse_options.sh
rvb_affix=_rvb_noise_$SNR
norvb_datadir=

echo $norvb_datadir
if [ $stage -le 0 ]; then
  echo "$0: the SNR level is $SNR"
  echo "$0: creating adding noise command"

  if [ ! -f ${norvb_datadir}${rvb_affix}${num_data_reps}_hires/feats.scp ]; then
    if [ ! -d "RIRS_NOISES/" ]; then
      # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
      echo "Download the noise data"
      #wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
      #unzip rirs_noises.zip
    fi
    utils/fix_data_dir.sh  ${norvb_datadir} 
    rvb_opts=()
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
    #rvb_opts+=(--rir-set-parameters "0.5, simulated_rirs_15mic_linear_array_0219/small/rir_list")
    #rvb_opts+=(--rir-set-parameters "0.5, simulated_rirs_15mic_linear_array_0219/medium/rir_list")
    rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)
# default snr "20:10:15:5:0"
    python steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --foreground-snrs $SNR \
      --background-snrs $SNR \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications ${num_data_reps} \
      --max-noises-per-minute 30 \
      --source-sampling-rate $sample_rate \
      ${norvb_datadir} ${norvb_datadir}${rvb_affix}${num_data_reps}

  fi
fi


if [ $stage -le 1 ]; then
  echo "generating Noise Wav"
  python make_wav.py --in_dir ${norvb_datadir}${rvb_affix}${num_data_reps} --out_dir ${norvb_datadir}${rvb_affix}${num_data_reps}/Ndata_${SNR}  
  find `pwd`/${norvb_datadir}${rvb_affix}${num_data_reps}/Ndata_${SNR}/ -name "*.raw" > pcm_${SNR}.scp
fi

