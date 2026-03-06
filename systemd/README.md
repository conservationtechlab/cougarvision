If you want cougarvision to run on reboots automatically, 
edit the below scripts to match your user/paths and run the following:

```
sudo cp cougarvision.service /etc/systemd/system/cougarvision.service
sudo systemctl daemon-reload
sudo systemctl start cougarvision.service
```

To check if it's running successfully:

```
sudo journalctl -u cougarvision.service
```
