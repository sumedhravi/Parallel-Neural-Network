
rm -rf cuda.json
nvcc cuda_perceptron_single_thread.cu -o cuda_perceptron_single_thread

OUTPUT_FILE_1="cuda.json"

echo "[" > $OUTPUT_FILE_1

FIRST_ITER="yes"
for inputs in {1..5}; do
  if [ "$FIRST_ITER" = "yes" ]; then
    FIRST_ITER="no"
  else
    echo "," >> $OUTPUT_FILE_1
  fi
  ./cuda_perceptron_single_thread $inputs >> $OUTPUT_FILE_1
done
echo "]" >> $OUTPUT_FILE_1
