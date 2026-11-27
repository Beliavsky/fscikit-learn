program reference_huber_invalid_driver
  use sklearn_common,only:dp
  use sklearn_linear_model,only:huber_regressor
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic,only:ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  type(huber_regressor)::model
  type(csr_matrix)::sparse
  real(dp),allocatable::x(:,:),y(:),weight(:),prediction(:)
  character(64)::representation,name,token
  real(dp)::value
  logical::column
  call get_command_argument(1,representation);call get_command_argument(2,name);call get_command_argument(3,token)
  read(token,*)value
  allocate(x(5,2));x(:,1)=[-2._dp,-1._dp,0._dp,1._dp,2._dp];x(:,2)=[1._dp,0._dp,1._dp,2._dp,-1._dp]
  y=[2._dp,-1._dp,3._dp,1._dp,0._dp];weight=spread(1._dp,1,5);column=.false.
  select case(name)
  case('epsilon');model%epsilon=value
  case('alpha');model%alpha=value
  case('tol');model%tol=value
  case('max_iter');model%max_iter=int(value)
  case('weight_nan');weight(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('weight_inf');weight(1)=ieee_value(0._dp,ieee_positive_inf)
  case('weight_negative');weight(1)=-1
  case('weight_all_negative');weight=-1
  case('weight_zero');weight=0
  case('weight_length');weight=[1._dp,2._dp]
  case('x_nan');x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  case('x_inf');x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  case('y_nan');y(1)=ieee_value(0._dp,ieee_quiet_nan)
  case('y_inf');y(1)=ieee_value(0._dp,ieee_positive_inf)
  case('empty_rows');x=x(:0,:);y=y(:0);weight=weight(:0)
  case('empty_columns');x=x(:,:0)
  case('target_column');column=.true.
  case('target_matrix')
    if(representation=='dense')then
      call model%fit(x,spread(y,2,2))
    else
      sparse=csr_from_dense(x);call model%fit(sparse,spread(y,2,2))
    end if
    print '(a)','FIT ACCEPTED';stop
  case('warm_wider','warm_narrower')
    call model%fit(x,y);model%warm_start=.true.
    if(name=='warm_wider')then
      x=reshape([x,x(:,1)],[5,3])
    else
      x=x(:,:1)
    end if
  end select
  sparse=csr_from_dense(x)
  if(representation=='dense')then
    if(column)then
      call model%fit(x,reshape(y,[size(y),1]),weight)
    else
      call model%fit(x,y,weight)
    end if
  else
    if(column)then
      call model%fit(sparse,reshape(y,[size(y),1]),weight)
    else
      call model%fit(sparse,y,weight)
    end if
  end if
  select case(name)
  case('query_width');x=x(:,:1)
  case('query_empty');x=x(:0,:)
  case('query_nan');x(1,1)=ieee_value(0._dp,ieee_quiet_nan)
  case('query_inf');x(1,1)=ieee_value(0._dp,ieee_positive_inf)
  end select
  if(index(name,'query_')==1)then
    if(representation=='dense')then
      prediction=model%predict(x)
    else
      sparse=csr_from_dense(x);prediction=model%predict(sparse)
    end if
  end if
  print '(a)','FIT ACCEPTED'
end program
