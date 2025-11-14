
class TFT(nn.Module):
    def __init__(self, lr, dropout, lstm_hidden, mha, hidden_c_size):
        super(TFT, self).__init__()

        self.tft_model = TemporalFusionTransformer.from_dataset(
            tr_dataset, # 모델이 사용할 피저 구조, 인코더-디코더 길이 등을 파악, TomeSeiresDataSet 객체
            hidden_size = lstm_hidden, # LSTM의 인코더-디코더 각 레이어의 은닉 차원 수, 클수록 과적합 가능성, 연산 비용 커짐
            attention_head_size = mha , #멀티헤드 셀픙 어텐션 헤드 개수
            dropout = dropout,
            hidden_continuous_size = hidden_c_size,  # 연속형 피처 처리용 작은 은닉층 크기, 모델 내부에서 GRN이 각 연속형 피처를 임베딩하는 차원
            output_size = 1, # 모델 최종 출력 차원
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            params = self.tft_model.parameters(), 
            lr = lr,
        )
        
    
        self.early_stop = early_stop(5, 0.0001)

    def smape(self, preds, trues, eps = 1e-8): 
        """
        Symmetric Mean Absolute Percentage Error (SMAPE) Loss
        """
        numerator = torch.abs(trues - preds)
        denominator = torch.abs(trues) + torch.abs(preds) + eps
        smape = 2.0 *numerator / denominator

        return torch.mean(smape)


    def train_model(self):
        best_loss_ever = float('inf')
        best_weight = None
        num_epochs = 50
        for epoch in range(num_epochs):
            self.tft_model.train()
            for batch in tqdm(tr_dataloader, total = len(tr_dataloader)):
                inputs, (targets, _) = batch
                
                inputs = {k : v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)

            
                self.optimizer.zero_grad()
                output = self.tft_model(inputs)  # model.predict() 메서드는 데이터로더를 입력으로 받
                y_preds = (output.prediction).squeeze(-1)


                loss = self.smape(y_preds, targets)
                
                loss.backward()
                self.optimizer.step()
                clip_grad_norm_(self.tft_model.parameters(), max_norm = 10)

            val_loss = self.validation()

            if val_loss < best_loss_ever:
                best_loss_ever = val_loss
                best_weight = copy.deepcopy(self.tft_model.state_dict())
                print(best_loss_ever)
                print("best Loss")
                print("===================================")

            if self.early_stop(val_loss):
                print(f"Early Stopping at epoch : {epoch + 1}")
                break

        if best_weight is not None:
            self.tft_model.load_state_dict(best_weight)
        else:
            print("Not Saved Params")

        return best_loss_ever

    def validation(self):
        self.tft_model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_inputs, (val_targets, _) = val_batch
                val_inputs = {k:v.to(device) for k, v in val_inputs.items()}
                val_targets = val_targets.to(device)

                val_output = self.tft_model(val_inputs)
                val_preds = (val_output.prediction).squeeze(-1)

                # val_total_loss += self.criterion(val_preds, val_targets).item()
                val_total_loss += self.smape(val_preds, val_targets).item()
               

        return val_total_loss / len(val_dataloader)
        # return val_total_loss 

    def test(self):
        self.tft_model.eval()
        all_preds = []
        with torch.no_grad():
            # test_preds = self.tft_model.predict(test_dataloader)
            for test_batch in test_dataloader:
                test_inputs, (test_targets, _) = test_batch
                test_inputs = {k:v.to(device) for k, v in test_inputs.items()}
                test_targets = test_targets.to(device)

                test_output = self.tft_model(test_inputs)
                test_preds = (test_output.prediction).squeeze(-1)
   
                all_preds.append(test_preds)
            
        return torch.cat(all_preds, dim = 0)
