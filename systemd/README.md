If you want cougarvision to run on reboots automatically, 
edit the below scripts to match your user/paths and run the following:

```
sudo chmod +x cougarvision.sh
sudo cp cougarvision.service /etc/systemd/system/cougarvision.service
sudo cp notify-service@cougarvision.service /etc/systemd/system/notify-service@cougarvision.service
sudo systemctl daemon-reload
sudo systemctl start cougarvision.service
```

To check if it's running successfully:

```
sudo journalctl -u cougarvision.service
```

'''
instuctions for setting up msmtp and email notifcations on failure
install msmtp:

sudo apt-get install msmtp


create configuation file in user home directory:

cat ~/.msmtprc
defaults
auth    on
tls_starttls off
tls    on
tls_trust_file    /etc/ssl/certs/ca-certificates.crt
logfile ~/.msmtp.log
account gmail
host	imap.gmail.com
port	465
from	bioreserve.cam@gmail.com
user	bioreserve.cam@gmail.com
password   bwdlcxasdytfekof

account default : gmail
EOF


before running cougarvision.service update systemd files:

sudo cp notify-service@cougarvision.service /etc/systemd/system/notify-service@cougarvision.service
sudo systemctl daemon-reload

'''