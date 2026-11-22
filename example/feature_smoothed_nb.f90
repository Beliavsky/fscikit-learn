program feature_smoothed_nb
  use sklearn, only: dp,multinomial_nb
  implicit none
  type(multinomial_nb) :: model
  real(dp) :: x(3,3),query(1,3)
  real(dp), allocatable :: probability(:,:)
  x(1,:)=[2._dp,0._dp,1._dp];x(2,:)=[1._dp,0._dp,2._dp];x(3,:)=[0._dp,3._dp,0._dp]
  model%alpha_per_feature=[0._dp,.2_dp,1._dp];model%force_alpha=.false.
  call model%fit(x,[-3,-3,9])
  query(1,:)=[0._dp,1._dp,1._dp];probability=model%predict_proba(query)
  print '(a,3f12.5)', 'Input smoothing: ',model%alpha_per_feature
  print '(a,2f14.8)', 'Floored-smoothing probabilities: ',probability
  print '(a,i4)', 'Predicted class: ',model%predict(query)
end program
