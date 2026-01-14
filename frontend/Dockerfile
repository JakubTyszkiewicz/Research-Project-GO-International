# Use Node 18
FROM node:18-alpine

WORKDIR /app

# Copy package json
COPY package.json .

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start dev server
CMD ["npm", "run", "dev", "--", "--host"]
