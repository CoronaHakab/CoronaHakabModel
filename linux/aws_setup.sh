link_base=https://raw.githubusercontent.com/CoronaHakab/CoronaHakabModel/develop/linux/
file_links=(deb_install_python3.8.sh deb_run.sh aws_cli.sh)

echo "alias py='python3.8'">>~/.bashrc
cat ~/.bashrc | grep "alias py"

SECONDS=0
for f in "${file_links[@]}"; do
	curl "$link_base$f" -o "$f"
	chmod 755 ./"$f"
	./"$f"
done
echo "setup took $SECONDS seconds"
