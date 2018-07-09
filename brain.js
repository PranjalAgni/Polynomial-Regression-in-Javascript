const tf = require('@tensorflow/tfjs');

//Equation: y = ax^2 + bx + c
//a,b,c are variables
//x is the number whose square we have to find
//y is the predixtion.
let a = tf.variable(tf.scalar(Math.random()));
let b = tf.variable(tf.scalar(Math.random()));
let c = tf.variable(tf.scalar(Math.random()));

//hyperparameters
let learning_rate = 0.05;
let optimizer = tf.train.adam(learning_rate);

//For training data: It pushes number 1 to 100 in an array
function generateXData() {
  xdata = [];
  for (let i=0; i<=100; i++) {
    xdata.push(i);
  }
  return xdata;
}


//For training data: It has all the squares of numbers in xdata
function generateYData() {
  ydata = [];
  for (let i=0; i<=100; i++) {
    ydata.push(i*i);
  }
  return ydata;
}


//converted array to a 1d tensor.
const xvals = tf.tensor1d(generateXData());
const yvals = tf.tensor1d(generateYData());

//Now we are normalizing the data.
//Normalizing the data means we are mapping the data 1 to 100 => 0 to 1
//This is helpfull to train large data.
const xmin = xvals.min(); //0
const xmax = xvals.max(); //100
const xdiff = xmax.sub(xmin); //100

function norm(x) {
  return x.sub(xmin).div(xdiff);
}

//We are mapping our y_vals according to range of x_vals because then it will form a relationship b/w both fields.

const xvals_norm = norm(xvals);
const yvals_norm = norm(yvals);

//Predict function.
function predict(x_arr) {
  //const x_tensor = tf.tensor1d(x_arr);

  const y_tensor =  x_arr.square().mul(a).add(x_arr.mul(b)).add(c);

  return y_tensor;
}

//Loss function ==> Minimizing loss.
function loss(prediction , label) {
  return prediction.sub(label).square().mean();
}


//Training the data set 25K iterations.
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

//Playing with it ...
//Predicting squares for few numbers.
// As we have normalized the data thats why I am giving input as input/100
predict(tf.tensor1d([0.04,0.05,5.25])).print();
//a,b,c are our variables with their best values after 25000 iterations.
a.print();
b.print();
c.print();
yvals_norm.print();


