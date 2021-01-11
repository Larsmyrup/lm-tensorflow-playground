const path = require("path")
const src = './src'
module.exports = {
  runtimeCompiler: true,
  configureWebpack: {
    resolve: {
      alias: {
        "@": path.join(__dirname, src),
      },
      extensions: ['.js', '.vue', '.json'],
    },
  },
}
