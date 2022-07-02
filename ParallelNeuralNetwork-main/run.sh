make clean
make

rm -rf sequential.json
rm -rf openMP.json

OUTPUT_FILE_1="sequential.json"
OUTPUT_FILE_2="openMP.json"
echo "[" > $OUTPUT_FILE_1
echo "[" > $OUTPUT_FILE_2

FIRST_ITER="yes"
for inputs in {1..10}; do
  if [ "$FIRST_ITER" = "yes" ]; then
    FIRST_ITER="no"
  else
    echo "," >> $OUTPUT_FILE_1
    echo "," >> $OUTPUT_FILE_2
  fi
  ./Perceptron $inputs 0 >> $OUTPUT_FILE_1
  ./Perceptron $inputs 1 >> $OUTPUT_FILE_2
done
echo "]" >> $OUTPUT_FILE_1
echo "]" >> $OUTPUT_FILE_2