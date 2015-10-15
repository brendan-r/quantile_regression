data {
  int<lower=0> N;
  real<lower=0, upper=1> p;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
  vector<lower=0>[N] w;
}
transformed parameters {
  real<lower=0> tau;
  vector[N] mu;
  vector[N] me;
  vector[N] pe;
  vector[N] pe2;

  tau <- pow(sigma, -2);
  mu  <- alpha + beta * x;
  me  <- (1 - 2 * p) / (p * (1 - p)) * w + mu;
  pe  <- (p * (1 - p) * tau) ./ (2 * w);

  for(n in 1:N)
    pe2[n] <- inv_sqrt(pe[n]);
}
model {
  # Priors
  alpha ~ normal(0, 2);
  beta  ~ normal(0, 2);
  sigma ~ cauchy(0, 2.5);

  # Data Augmentation
  w     ~ exponential(tau);

  # The model
  y     ~ normal(me, pe2);
}
