link_base=https://raw.githubusercontent.com/CoronaHakab/CoronaHakabModel/develop/linux/
file_links=(deb_install_python3.8.sh aws_cli.sh deb_run.sh)

SECONDS=0
for f in "${file_links[@]}"; do
	curl "$link_base$f" -o "$f"
	chmod 755 ./"$f"
	./"$f"
done
echo "setup took $SECONDS seconds"
