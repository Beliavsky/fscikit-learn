program unsmoothed_categories
  use sklearn, only: dp,categorical_nb
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(categorical_nb) :: model
  real(dp) :: x(4,2),query(1,2)
  real(dp), allocatable :: probability(:,:)
  x(1,:)=[0._dp,0._dp]; x(2,:)=[0._dp,1._dp]
  x(3,:)=[1._dp,0._dp]; x(4,:)=[1._dp,1._dp]
  model%alpha=0; model%min_categories=3
  call model%fit(x,[0,0,1,1])
  query=2; probability=model%predict_proba(query)
  print '(a,l1)', 'Unseen unsmoothed category is undefined: ',all(ieee_is_nan(probability))
  model%force_alpha=.false.; call model%fit(x,[0,0,1,1])
  probability=model%predict_proba(query)
  print '(a,2f14.8)', 'Floored smoothing probabilities: ',probability
end program
