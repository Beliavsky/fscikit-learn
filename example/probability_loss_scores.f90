program probability_loss_scores
  use sklearn, only: dp,log_loss,d2_log_loss_score
  implicit none
  integer :: y(4)=[0,1,1,0]
  real(dp) :: probability(4)=[.1_dp,.8_dp,.7_dp,.2_dp],weights(4)=[1._dp,2._dp,1._dp,1._dp]
  print *, 'Mean binary log loss:',log_loss(y,probability)
  print *, 'Weighted summed log loss:',log_loss(y,probability,sample_weight=weights,normalize=.false.)
  print *, 'Fraction of log loss explained:',d2_log_loss_score(y,probability)
end program
