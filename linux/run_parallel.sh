!/bin/bash
num_of_runs="$1"
shift

for ((i=0; i< $num_of_runs - 1; i++))
do
   python3.8 ./src/corona_hakab_model/main.py $@ > "${i}_log.tmp" 2>&1 &
   echo "Running iteration ${i} in the background"
done
echo "Running iteration ${i}, it might take a while..."
python3.8 ./src/corona_hakab_model/main.py $@ 2>&1 | grep "OUTPUT FOLDER"
echo "Finished iteration ${1}"
echo "Other outputs folders:"
for ((i=0; i< $num_of_runs - 1; i++))
do
   grep "OUTPUT FOLDER" "${i}_log.tmp"
   rm -f "${i}_log.tmp"
done
echo Done
