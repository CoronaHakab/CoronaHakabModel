WORKDIR=~
PROJDIR=${WORKDIR}/proj
cd $PROJDIR
for ((i=0; i< $1 - 1; i++))
do
   python3.8 ./src/corona_hakab_model/main.py all > "${i}_log.tmp" 2>&1 &
   echo "Running iteration ${i+1} in the background"
done
echo "Running iteration ${1}, it might take a while..."
python3.8 ./src/corona_hakab_model/main.py all> "${1}_log.tmp" 2>&1
echo "Finished iteration ${1}"
echo "Other outputs folders:"
for ((j=0; j< $1; j++))
do
   echo "For iteration ${j+1}:"
   grep "OUTPUT FOLDER" "${j}_log.tmp"
   rm -f "${j}_log.tmp"
done
echo Done