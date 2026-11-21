program multiclass_reductions
  use sklearn, only: dp,one_vs_rest_classifier,one_vs_one_classifier,output_code_classifier,logistic_regression
  implicit none
  type(one_vs_rest_classifier) :: rest
  type(one_vs_one_classifier) :: pairs
  type(output_code_classifier) :: code
  type(logistic_regression) :: base
  real(dp) :: x(12,2),query(3,2)
  integer :: y(12),i
  x(:,1)=[(real(i-6,dp),i=1,12)]; x(:,2)=[(real(mod(i,3),dp),i=1,12)]
  y=[-7,-7,-7,-7,4,4,4,4,19,19,19,19]; query=x([2,6,11],:)
  call rest%fit(x,y,base); call pairs%fit(x,y,base)
  code%random_state=42; code%code_size=2; call code%fit(x,y,base)
  print *, 'OVR predictions:',rest%predict(query)
  print *, 'OVR probabilities:',rest%predict_proba(query)
  print *, 'OVO predictions:',pairs%predict(query)
  print *, 'Output-code predictions:',code%predict(query)
  print *, 'Output-code book shape:',shape(code%code_book_)
end program
