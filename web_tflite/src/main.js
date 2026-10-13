import { pipeline } from '@huggingface/transformers';

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
    const model = await tflite.loadTFLiteModel('./SP-all-MiniLM-L6-v2.tflite');
    console.log(model);

    btn.addEventListener('click', async () => {
        const sentences = [input.value];
        const output = await extractor(sentences, { pooling: 'mean', normalize: true });
        const outputList = output.tolist();
        const outputTensor = await model.predict(tf.tensor2d(outputList));
        const probabilitiesTensor = await tf.sigmoid(outputTensor);
        const probabilities = await probabilitiesTensor.array();
        console.log(probabilities);

        // Update the UI with the results
        if (probabilities && probabilities.length > 0) {
            const scores = probabilities[0]; // Assuming batch size of 1
            labels.forEach((label, index) => {
                if (scoreElements[label]) {
                    scoreElements[label].textContent = scores[index].toFixed(4); // Format to 4 decimal places
                }
            });
        }
    });
})();
