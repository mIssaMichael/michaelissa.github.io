data {
  int<lower=0> N;              // Number of players
  array[N] int<lower=0> Tr;     // Trials for each player
  array[N] int<lower=0> y;      // Successes for each player
}

parameters {
  vector[N] alpha;              // Logit-transformed chance of success for each player
}

model {
  alpha ~ normal(0, 10);       // Prior for alpha (logit scale)

  for (n in 1:N) {
    target += binomial_logit_lpmf(y[n] | Tr[n], alpha[n]);
  }
}

generated quantities {
   array[N] int y_pred;          // Predicted number of successes
   array[N] real p_hat_pred;              // Mean predicted probability of success

   for (n in 1:N) {
     y_pred[n] = binomial_rng(Tr[n], inv_logit(alpha[n]));
     p_hat_pred[n] = inv_logit(alpha[n]);
   }

}
