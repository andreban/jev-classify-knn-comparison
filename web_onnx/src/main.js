import { pipeline } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

const input = document.getElementById('input');
const btn = document.getElementById('btn');
const scoreElements = {
    toxic: document.getElementById('toxic-score'),
    severe_toxic: document.getElementById('severe_toxic-score'),
    obscene: document.getElementById('obscene-score'),
    threat: document.getElementById('threat-score'),
    insult: document.getElementById('insult-score'),
    identity_hate: document.getElementById('identity_hate-score')
};

// Define the labels in the order the model outputs them
const labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'];

(async () => {
    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    const model = await ort.InferenceSession.create('./SP-all-MiniLM-L6-v2.onnx', {
        executionProviders: ['wasm'],
    });
    console.log(model);

    btn.addEventListener('click', async () => {
        const sentences = [input.value];
        const output = await extractor(sentences, { pooling: 'mean', normalize: true });
        const outputTensor = await model.run({x: output});
        const probabilities = sigmoid(outputTensor.linear_2.data);

        // Update the UI with the results
        if (probabilities && probabilities.length > 0) {
            labels.forEach((label, index) => {
                if (scoreElements[label]) {
                    console.log(label, probabilities[index]);
                    scoreElements[label].textContent = probabilities[index].toFixed(4); // Format to 4 decimal places
                }
            });
        }
    });
})();

function sigmoid(xs) {
    return xs.map(x=> 1 / (1 + Math.exp(-x)))
}
