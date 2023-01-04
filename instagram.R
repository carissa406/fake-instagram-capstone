FLAGS <- flags(
  flag_numeric("nodes1", 32),
  flag_numeric("nodes2", 32),
  flag_numeric("batch_size", 32),
  flag_numeric("epochs", 100)
) 

model = keras_model_sequential()%>%
  layer_dense(units = FLAGS$nodes1, activation = "relu", input_shape = dim(train.nn)[2])%>%
  layer_dense(units = FLAGS$nodes2, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "sgd",
  loss = "binary_crossentropy",
  metrics = "accuracy")

model %>% fit(
  train.nn, log(train.labels), 
  epochs = FLAGS$epochs, 
  batch_size= FLAGS$batch_size,
  validation_data=list(val.norm_nn, log(val.labels)))