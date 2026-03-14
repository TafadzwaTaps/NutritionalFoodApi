// vite.config.js
export default {
  server: {
    proxy: {
      '/analyze': {
        target: 'https://nutritionalfoodapi-c0ye.onrender.com',
        changeOrigin: true,
        secure: true,
        rewrite: path => path.replace(/^\/analyze/, '/analyze'),
      }
    }
  }
};
