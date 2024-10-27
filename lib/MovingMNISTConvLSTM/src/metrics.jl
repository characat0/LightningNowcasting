

matches(y_pred, y_true) = sum((y_pred .> 0.5f0) .== (y_true .> 0.5f0))
TP(y_pred, y_true) = sum(@. (y_true > 0.5f0) * (y_pred > 0.5f0))
TN(y_pred, y_true) = sum(@. (y_true < 0.5f0) * (y_pred < 0.5f0))

FN(y_pred, y_true) = sum(@. (y_true > 0.5f0) * (y_pred < 0.5f0))
FP(y_pred, y_true) = sum(@. (y_true < 0.5f0) * (y_pred > 0.5f0))

f1(y_pred, y_true) = 2*TP(y_pred, y_true)/(2*TP(y_pred, y_true) + FP(y_pred, y_true) + FN(y_pred, y_true))


accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
