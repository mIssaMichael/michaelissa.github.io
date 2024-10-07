data {
  int<lower=0> N;              // Number of players
  array[N] int<lower=0> Tr;     // Trials for each player
  array[N] int<lower=0> y;      // Successes for each player
}

parameters {
  vector[N] alpha;              // Logit-transformed chance of success for each player
  real mu;
  real<lower=0> sigma; 
}

model {
  sigma ~ normal(0, 1);   // Hyperprior
  mu ~ normal(0.3, sigma);    // Hyperprior
  alpha ~ normal(mu, sigma);

  for (n in 1:N) {
    target += binomial_logit_lpmf(y[n] | Tr[n], mu + alpha[n] * sigma);
  }
}

generated quantities {
   array[N] int y_pred;          // Predicted number of successes
   array[N] real p_hat_pred;              // Mean predicted probability of success


   for (n in 1:N) {
     y_pred[n] = binomial_rng(Tr[n], inv_logit(mu + alpha[n] * sigma));
     p_hat_pred[n] = inv_logit(mu + alpha[n] * sigma);
   }

}
