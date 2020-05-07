#!/bin/bash
num_of_runs=$1
circle_consts=$2
matrix_consts=$3
simulation_consts=$4
TIMESTAMP=`date --utc -d "+3 hours" +_%d%m%Y_%H_%M`
output_folder=$5$TIMESTAMP


python3.8 ./src/corona_hakab_model/main.py generate -c $circle_consts -m $matrix_consts -o $output_folder
cp $circle_consts $output_folder
cp $matrix_consts $output_folder
cp $simulation_consts $output_folder


for ((i=0; i< $num_of_runs - 1; i++))
do
   python3.8 ./src/corona_hakab_model/main.py simulate --population-data "${output_folder}/population_data.pickle" --matrix-data "${output_folder}/matrix_data.parasymbolic" --connection_data "${output_folder}/connection_data.pickle" -s $simulation_consts  --output "${output_folder}/${i}" > "${i}_log.tmp" 2>&1 &
   echo "Running iteration ${i} in the background"
done
echo "Running iteration ${i}, it might take a while..."
python3.8 ./src/corona_hakab_model/main.py simulate --population-data "${output_folder}/population_data.pickle" --matrix-data "${output_folder}/matrix_data.parasymbolic" --connection_data "${output_folder}/connection_data.pickle" -s $simulation_consts --output "${output_folder}/${i}"
echo "Finished iteration ${1}"
echo "Other outputs folders:"
for ((i=0; i< $num_of_runs - 1; i++))
do
   grep "OUTPUT FOLDER" "${i}_log.tmp"
   rm -f "${i}_log.tmp"
done
echo Done
