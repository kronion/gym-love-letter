const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin')

module.exports = {
  context: path.resolve(__dirname, 'src'),
  entry: './index.tsx',
  mode: 'development',
  devtool: 'source-map',
  module: {
    rules: [
      {
        test: /\.(j|t)sx?$/i,
        use: 'babel-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.scss$/i,
        use: [
          'style-loader',
          'css-modules-typescript-loader',
          'css-loader',
          'sass-loader'
        ]
      },
    ],
  },
  resolve: {
    extensions: [ '.js', '.jsx', '.ts', '.tsx' ],
  },
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'Custom template',
      template: 'index.html',
      base: "/static/"
    })
  ],
  watch: true
};
