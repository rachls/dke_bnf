#!/bin/bash
#SBATCH --job-name=arm_download
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=arm_download_%j.out
#SBATCH --error=arm_download_%j.err

set -uo pipefail

ORDER_ID="286044"
ORDER_USER="stephanr1"
EXPECTED_FILES=1978
DEST="/stor/soteria/hydro/private/rls134/data/dke_bnf/ARM/raw"
HOSTS=("ftp.archive.arm.gov" "ftp2.archive.arm.gov")

mkdir -p "${DEST}"
cd "${DEST}" || exit 1

echo "Job ${SLURM_JOB_ID:-interactive} started on $(hostname) at $(date)"
echo "Destination: ${DEST}"

download_with_lftp() {
    local host="$1"
    lftp -u anonymous,anonymous \
        -e "set net:max-retries 5; \
            set net:reconnect-interval-base 10; \
            set net:timeout 30; \
            mirror --continue --parallel=4 --verbose \
                /${ORDER_USER}/${ORDER_ID} ${DEST}; \
            quit" \
        "ftp://${host}"
}

download_with_wget() {
    local host="$1"
    wget --recursive --no-parent --no-host-directories --cut-dirs=2 \
        --continue --tries=5 --waitretry=10 --timeout=30 \
        --reject "index.html*" \
        "ftp://${host}/${ORDER_USER}/${ORDER_ID}/"
}

for host in "${HOSTS[@]}"; do
    echo "Attempting download from ${host}"
    if command -v lftp > /dev/null 2>&1; then
        download_with_lftp "${host}"
    else
        download_with_wget "${host}"
    fi

    n_files=$(find "${DEST}" -type f | wc -l)
    echo "Files present: ${n_files} of ${EXPECTED_FILES}"
    if [ "${n_files}" -ge "${EXPECTED_FILES}" ]; then
        break
    fi
done

n_files=$(find "${DEST}" -type f | wc -l)
total_size=$(du -sh "${DEST}" | cut -f1)
echo "Final count: ${n_files} files, ${total_size} (expected ${EXPECTED_FILES}, ~1.6 GB)"

if [ "${n_files}" -ne "${EXPECTED_FILES}" ]; then
    echo "ERROR: file count mismatch. Resubmit to resume." >&2
    exit 1
fi

echo "Download complete at $(date)"
