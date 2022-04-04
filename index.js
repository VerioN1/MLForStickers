const express = require("express");
const app = express();
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const Jimp = require('jimp');
const multer = require("multer");
const path = require('path')
const bodyParser = require('body-parser');

app.use(express.static("./static"));
app.use(bodyParser.urlencoded({ extended: false}));
app.use(bodyParser.json({limit: '50mb'}));


const storage = multer.diskStorage({
    destination: "./uploads/",
    filename: function (req, file, cb) {
        cb(null,  "SomeImage" + "." + file.originalname.split(".").pop());
    },
});
const memoryStorage = multer({
    storage: multer.memoryStorage(),
});
const diskStorage = multer({ storage: storage });

// const IMAGE_FILE_PATH = path.resolve(__dirname, './assets/photoTest.jpg')
const labels = require(`${__dirname}/metadata.json`).labels;


const PORT = process.env.PORT || 8080

const main = async(imageData) => {
    let IMG;
    if(imageData.indexOf('base64') !== -1) {
        IMG = await Jimp.read(Buffer.from(imageData.split(",")[1], 'base64'))
    } else {
        IMG = await Jimp.read(Buffer.from(imageData, 'base64'))
    }
    const model = path.resolve(__dirname, './model.json');
    const handler = tfn.io.fileSystem(model)
    const TrainedModel = await tfn.loadLayersModel(handler);
    TrainedModel.summary();
    // IMG = await Jimp.read(IMAGE_FILE_PATH);
    IMG.cover(224, 224, Jimp.HORIZONTAL_ALIGN_CENTER | Jimp.VERTICAL_ALIGN_MIDDLE);
    const fineImage = prepareForTensor(IMG)
    const result = {}
    const predictions = await TrainedModel.predict(fineImage).dataSync();
    for (let i = 0; i < predictions.length; i++) {
        const label = labels[i];
        const probability = predictions[i];
        console.log(`${label}: ${probability}`);
        result[label] = probability;
    }
    return result;
}

app.listen(PORT, () => {
    console.log(
        `Server is active on ${PORT}`
    );
});
app.post('/', async (req, res) => {
    const imageData = req.body.imgData;
    const result = await main(imageData);
    res.send(result);
});
app.get('/', (req, res) => {
    res.send('App is running');
});


const prepareForTensor = (image) => {
    const NUM_OF_CHANNELS = 3;
    let values = new Float32Array(224 * 224 * NUM_OF_CHANNELS);

    let i = 0;
    image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
        const pixel = Jimp.intToRGBA(image.getPixelColor(x, y));
        pixel.r = pixel.r / 127.0 - 1;
        pixel.g = pixel.g / 127.0 - 1;
        pixel.b = pixel.b / 127.0 - 1;
        pixel.a = pixel.a / 127.0 - 1;
        values[i * NUM_OF_CHANNELS] = pixel.r;
        values[i * NUM_OF_CHANNELS + 1] = pixel.g;
        values[i * NUM_OF_CHANNELS + 2] = pixel.b;
        i++;
    });
    const outShape = [224, 224, NUM_OF_CHANNELS];
    let img_tensor = tf.tensor3d(values, outShape, 'float32');
    return img_tensor.expandDims(0);
}
