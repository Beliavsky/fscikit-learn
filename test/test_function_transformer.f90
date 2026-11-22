program test_function_transformer
  use sklearn_function_transformer
  use sklearn_common, only: dp,require
  use sklearn_feature_extraction, only: feature_string
  implicit none
  type(function_transformer) :: model,copy
  type(feature_string), allocatable :: names(:)
  real(dp) :: x(4,2)
  real(dp), allocatable :: z(:,:),back(:,:)
  x=reshape([1._dp,2._dp,3._dp,4._dp,5._dp,6._dp,7._dp,8._dp],[4,2])
  z=model%transform(x)
  call require(all(z==x),'FunctionTransformer identity needs no fit')
  model%func=>affine;model%inverse_func=>undo_affine;model%validate=.true.
  model%feature_names_out='one-to-one'
  z=model%fit_transform(x);back=model%inverse_transform(z)
  call require(model%inverse_checked_.and.model%inverse_consistent_,'FunctionTransformer inverse check')
  call require(maxval(abs(back-x))<1.e-14_dp,'FunctionTransformer affine roundtrip')
  names=model%get_feature_names_out()
  call require(names(1)%text=='x0'.and.names(2)%text=='x1','FunctionTransformer generated names')
  copy=model;copy%inverse_func=>affine;call copy%fit(x)
  call require(.not.copy%inverse_consistent_,'FunctionTransformer records noninverse callbacks')
  call require(model%inverse_consistent_,'FunctionTransformer copy is independent')
  model%func=>expand;model%inverse_func=>contract
  z=model%fit_transform(x);back=model%inverse_transform(z)
  call require(all(shape(z)==[4,4]).and.all(back==x),'inverse accepts changed forward width')
  model%validate=.false.;nullify(model%func,model%inverse_func)
  call model%fit(x(:,:1))
  call require(model%n_features_in_==1,'unvalidated fit still resets feature metadata')
  names=model%get_feature_names_out()
  call require(size(names)==1.and.names(1)%text=='x0','unvalidated fit has generated names')
  print *, 'FunctionTransformer: PASS'
contains
  function affine(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=2*a+3
  end function
  function undo_affine(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=(a-3)/2
  end function
  function expand(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    allocate(b(size(a,1),2*size(a,2)));b(:,:size(a,2))=a;b(:,size(a,2)+1:)=a
  end function
  function contract(a) result(b)
    real(dp), intent(in) :: a(:,:)
    real(dp), allocatable :: b(:,:)
    b=a(:,:size(a,2)/2)
  end function
end program
