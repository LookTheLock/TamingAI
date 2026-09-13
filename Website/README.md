# TamingAI

[![Hugo](https://img.shields.io/badge/Hugo-%5E0.158.0-blue.svg)](https://gohugo.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An educational project about the technology of neural networks powered by [Hugo](https://github.com/gohugoio/hugo).

**[Demo →](https://taming-ai.net/)**

Consisting of two chapters, it covers the building blocks of a typical machine learning algorithm and explains how to apply them to a real project.

We also include our [own](https://github.com/LookTheLock/NeuralNetwork) simple neural network that is made to classify handwritten digits using the [MNIST](https://de.wikipedia.org/wiki/MNIST-Datenbank) dataset.

The algorithm itself is written from the ground up in `C++`. Therefore, we did **not** use any external libraries to build neural networks such as [OpenNN](https://www.opennn.net/), [Dlib](https://dlib.net/) etc.

TamingAI is ideal for people who don't only want to learn about how to use AI, but also **how it works** under the hood.

Learn more and contribute on GitHub: [NeuralNetwork](https://github.com/LookTheLock/NeuralNetwork).

## Quick Start

### Prerequisites
- Hugo >= **0.158.0** ([Download Hugo](https://gohugo.io/installation/))
- Git

### Try It Out
```bash
# Clone the repository
git clone https://github.com/LookTheLock/TamingAI
cd TamingAI/Website

# Start the development server
hugo server

# Open http://localhost:1313/ in your browser
```


## Contributing
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments
TamingAI makes use of a variety of open source projects including:

* https://github.com/phoebetronic/mnist
* https://github.com/gohugoio/hugo
* https://github.com/KaTeX/KaTeX
* https://github.com/vincentdoerig/latex-css
* https://github.com/hugotex-dev/HugoTeX
* https://reintech.io/blog/custom-drawing-application-html5-canvas-javascript
