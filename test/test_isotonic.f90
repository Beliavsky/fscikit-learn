program test_isotonic
  use sklearn
  use sklearn_common, only: require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  implicit none
  type(isotonic_regression) :: model
  real(dp) :: y(7)=[3._dp,1._dp,4._dp,2._dp,7._dp,6._dp,9._dp]
  real(dp) :: w(7)=[1._dp,2._dp,1._dp,3._dp,2._dp,1._dp,4._dp],x(7),residual(7)
  real(dp), allocatable :: z(:),again(:),query(:)
  integer :: i
  x=[(real(i,dp),i=1,7)]; z=isotonic_fit(y,w)
  call require(all(z(2:)>=z(:6)),'PAVA monotone fit')
  residual=w*(y-z)
  call require(abs(sum(residual))<1.e-12_dp,'PAVA weighted residual sum')
  do i=1,6
    call require(sum(residual(:i))>=-1.e-12_dp,'PAVA dual feasibility')
    call require(abs(sum(residual(:i))*(z(i+1)-z(i)))<1.e-12_dp,'PAVA complementary slackness')
  end do
  again=isotonic_fit(y,w*1.e150_dp)
  call require(maxval(abs(z-again))<1.e-12_dp,'PAVA invariant to weight scaling')
  call model%fit(x,y,w); again=model%predict(x)
  call require(maxval(abs(z-again))<1.e-12_dp,'isotonic interpolation at training points')
  query=model%predict([0._dp,8._dp]); call require(all(ieee_is_nan(query)),'isotonic default outside NaNs')
  model%out_of_bounds='clip'; query=model%predict([0._dp,8._dp])
  call require(maxval(abs(query-[z(1),z(7)]))<1.e-12_dp,'isotonic endpoint clipping')
  model%y_min=2; model%y_max=6; z=model%fit_transform(x,y,w)
  call require(all(z>=2).and.all(z<=6),'isotonic bounded fit')
  model%y_min=-huge(1._dp); model%y_max=huge(1._dp); model%auto_increasing=.true.
  z=model%fit_transform(x,-x)
  call require(.not.model%increasing_.and.maxval(abs(z+x))<1.e-12_dp,'automatic decreasing direction')
  call require(check_increasing([1._dp,1._dp,2._dp],[3._dp,3._dp,4._dp]),'Spearman tied ranks')
  call require(.not.check_increasing([1._dp,1._dp],[2._dp,3._dp]),'undefined Spearman direction')
  x=[1._dp,1._dp,2._dp,2._dp,3._dp,4._dp,100._dp]
  y=[0._dp,2._dp,3._dp,5._dp,6._dp,8._dp,-100._dp]; w=1; w(7)=0
  model%auto_increasing=.false.; model%increasing=.true.; call model%fit(x,y,w)
  query=model%predict([1._dp,1.5_dp,2._dp,4._dp,100._dp])
  call require(maxval(abs(query-[1._dp,2.5_dp,4._dp,8._dp,8._dp]))<1.e-12_dp, &
    'isotonic duplicate weighted means and zero weight removal')
  call require(abs(model%score([1._dp,2._dp,3._dp,4._dp],[1._dp,4._dp,6._dp,8._dp])-1)<1.e-12_dp, &
    'isotonic R2 score')
  call model%fit([0._dp,0.5e-15_dp,2.e-15_dp],[1._dp,3._dp,5._dp])
  call require(size(model%x_thresholds_)==2,'isotonic near-duplicate tolerance')
  query=model%predict([0._dp,2.e-15_dp])
  call require(maxval(abs(query-[2._dp,5._dp]))<1.e-12_dp,'isotonic near-duplicate mean')
  print *, 'isotonic regression: PASS'
end program
