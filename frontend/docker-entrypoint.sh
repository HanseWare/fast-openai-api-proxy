#!/bin/sh
# Fill the template with the environment variable or a default
envsubst '${BACKEND_URL}' < /usr/share/nginx/html/config.template.js > /usr/share/nginx/html/config.js

# Start Nginx
nginx -g 'daemon off;'