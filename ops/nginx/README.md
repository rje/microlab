# Microlab Console nginx

This directory stores nginx templates for exposing the local Microlab Console.

The app itself should continue to bind only to localhost:

```bash
systemctl --user status microlab-site
curl http://127.0.0.1:8765/
```

## Install

```bash
cd /home/rje/src/python/microlab
sudo cp ops/nginx/microlab.rje.ai.conf /etc/nginx/sites-available/microlab.rje.ai
sudo ln -s /etc/nginx/sites-available/microlab.rje.ai /etc/nginx/sites-enabled/microlab.rje.ai
sudo nginx -t
sudo systemctl reload nginx
```

Then immediately issue the certificate and let Certbot update the file:

```bash
sudo certbot --nginx -d microlab.rje.ai
sudo nginx -t
sudo systemctl reload nginx
```

## Basic Auth Before Public Use

Keep the site unauthenticated only long enough to bootstrap Certbot. Before
sharing the URL broadly, create an htpasswd file and uncomment the `auth_basic`
lines in `/etc/nginx/sites-available/microlab.rje.ai`:

```bash
sudo apt-get install apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd-microlab rje
sudo nginx -t
sudo systemctl reload nginx
```
