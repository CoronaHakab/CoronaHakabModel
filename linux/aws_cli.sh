sudo apt install -y unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
alias aws=/usr/local/bin/aws
aws --version

echo "to configure aws run the follwing command:"
echo "aws configure"
