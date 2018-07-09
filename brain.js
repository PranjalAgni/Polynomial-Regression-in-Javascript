const tf = require('@tensorflow/tfjs');


let a = tf.variable(tf.scalar(Math.random()));
let b = tf.variable(tf.scalar(Math.random()));
let c = tf.variable(tf.scalar(Math.random()));

let learning_rate = 0.05;
let optimizer = tf.train.adam(learning_rate);
function generateXData() {
  xdata = [];
  for (let i=0; i<=100; i++) {
    xdata.push(i);
  }
  return xdata;
}

function generateYData() {
  ydata = [];
  for (let i=0; i<=100; i++) {
    ydata.push(i*i);
  }
  return ydata;
}


const xvals = tf.tensor1d(generateXData());
const yvals = tf.tensor1d(generateYData());


const xmin = xvals.min(); //0
const xmax = xvals.max(); //100
const xdiff = xmax.sub(xmin); //100

function norm(x) {
  return x.sub(xmin).div(xdiff);
}

const xvals_norm = norm(xvals);
const yvals_norm = norm(yvals);


function predict(x_arr) {
  //const x_tensor = tf.tensor1d(x_arr);

  const y_tensor =  x_arr.square().mul(a).add(x_arr.mul(b)).add(c);

  return y_tensor;
}

function loss(prediction , label) {
  return prediction.sub(label).square().mean();
}

for (let i=0; i<25000; i++) {
  optimizer.minimize(function() {
    let inside =  loss(predict(xvals_norm),yvals_norm);
    inside.print();
    console.log('A = ');
    a.print();
    console.log('B = ');
    b.print();
    console.log('C = ');
    c.print();
    return inside;
  });

}
predict(tf.tensor1d([0.04,0.05,5.25])).print();
a.print();
b.print();
c.print();
yvals_norm.print();


