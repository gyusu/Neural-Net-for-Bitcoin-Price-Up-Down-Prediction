
epsilon = 1e-8

class STATS:

    def __init__(self, true_data, predicted_data, threshold=0.5):
        self.true_data = true_data
        self.pred_data = predicted_data
        self.threshold = threshold
        self.TP, self.FP, self.TN, self.FN = (0, 0, 0, 0)
        self.ignored = 0

        self._count_conditions()
        self._calc_stats()

    def _count_conditions(self):

        for i in range(len(self.true_data)):
            if self.true_data[i] == 1:
                if self.pred_data[i] >= self.threshold:
                    self.TP += 1
                elif self.pred_data[i] < 1 - self.threshold:
                    self.FN += 1
                else:
                    self.ignored += 1
            else:
                if self.pred_data[i] >= self.threshold:
                    self.FP += 1
                elif self.pred_data[i] < 1 - self.threshold:
                    self.TN += 1
                else:
                    self.ignored += 1

    def _calc_stats(self):
        self.total_condition = self.TP + self.FP + self.TN + self.FN
        self.accuracy = (self.TP + self.TN) / (self.total_condition + epsilon)
        self.precision = self.TP / (self.TP + self.FP + epsilon)
        self.recall = self.TP / (self.TP + self.FN + epsilon)
        self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall + epsilon)

    def print_stats(self):

        print('----------[Confusion Matrix]----------')
        print('> total_data={:}, threshold={}, ignored={}, total_condition={}'.format(
            len(self.true_data), self.threshold, self.ignored, self.total_condition))
        print('> TP:{}, FP:{}, TN:{}, FN:{}'.format(self.TP, self.FP, self.TN, self.FN))
        print('> Accuracy:{:.2}, Precision:{:.2}, Recall:{:.2}, F1:{:.2}'.format(self.accuracy, self.precision, self.recall, self.F1))
