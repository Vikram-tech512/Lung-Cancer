# web/Dockerfile – Vite React UI
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 5173
# Redirect nginx to port 5173
RUN sed -i 's/listen       80;/listen       5173;/' /etc/nginx/conf.d/default.conf
CMD ["nginx", "-g", "daemon off;"]
