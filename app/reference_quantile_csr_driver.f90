program reference_quantile_csr_driver
  use sklearn_quantile,only:quantile_regressor
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common,only:dp
  implicit none
  type(quantile_regressor)::model
  type(csr_matrix)::s,query
  real(dp),allocatable::x(:,:),y(:),w(:),score_weight(:)
  integer::n,p,mode,intercept,i,input,output
  character(1024)::path
  call get_command_argument(1,path);open(newunit=input,file=trim(path),status='old')
  read(input,*)n,p,mode,intercept,model%quantile,model%alpha
  allocate(x(n,p),y(n),w(n));model%fit_intercept=intercept==1
  do i=1,n
    read(input,*)x(i,:),y(i),w(i)
  end do
  close(input)
  if(mode>=5)then
    if(mode>=10)then
      s=raw_csr(x);query=raw_csr(x+.137_dp)
    else
      s=csr_from_dense(x,store_zeros=.true.);query=csr_from_dense(x+.137_dp)
    end if
  end if
  select case(mode)
  case(0);call model%fit(x,y)
  case(1);call model%fit(x,y,w)
  case(2);call model%fit(x,y,w(1))
  case(3);call model%fit(x,reshape(y,[n,1]),w)
  case(4);call model%fit(x,reshape(y,[n,1]),w(1))
  case(5);call model%fit(s,y)
  case(6,10);call model%fit(s,y,w)
  case(7);call model%fit(s,y,w(1))
  case(8,11);call model%fit(s,reshape(y,[n,1]),w)
  case(9);call model%fit(s,reshape(y,[n,1]),w(1))
  end select
  score_weight=w;if(sum(w)==0)score_weight=1
  call get_command_argument(2,path);open(newunit=output,file=trim(path),status='replace')
  call vector('coef',model%coef_);call vector('intercept',[model%intercept_])
  if(mode>=5)then
    call vector('predict',model%predict(s));call vector('query',model%predict(query))
    if(mode==8.or.mode==9.or.mode==11)then
      call vector('score',[model%score(s,reshape(y,[n,1]),score_weight)])
    else
      call vector('score',[model%score(s,y,score_weight)])
    end if
  else
    call vector('predict',model%predict(x));call vector('query',model%predict(x+.137_dp))
    if(mode==3.or.mode==4)then
      call vector('score',[model%score(x,reshape(y,[n,1]),score_weight)])
    else
      call vector('score',[model%score(x,y,score_weight)])
    end if
  end if
  call vector('dual',model%dual_coef_);call vector('diagnostics',[model%objective_,model%dual_gap_,model%optimality_])
  call vector('state',real([model%n_iter_,model%n_features_in_,merge(1,0,model%converged_)],dp));close(output)
contains
  function raw_csr(a)result(result)
    real(dp),intent(in)::a(:,:)
    type(csr_matrix)::result
    integer::r,j,k,ptr(size(a,1)+1),col(3*size(a))
    real(dp)::data(3*size(a))
    k=1;ptr(1)=1
    do r=1,size(a,1)
      do j=size(a,2),1,-1
        col(k:k+2)=j;data(k:k+2)=[.6_dp*a(r,j),.4_dp*a(r,j),0._dp];k=k+3
      end do
      ptr(r+1)=k
    end do
    result=csr_from_arrays(size(a,1),size(a,2),ptr,col,data)
  end function
  subroutine vector(name,a)
    character(*),intent(in)::name
    real(dp),intent(in)::a(:)
    write(output,'(a,1x,*(es26.17e3,1x))')name,a
  end subroutine
end program
