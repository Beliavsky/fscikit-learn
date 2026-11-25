program reference_kernel_ridge_driver
  use sklearn_common,only:dp,require
  use sklearn_kernel_ridge,only:kernel_ridge
  use sklearn_sparse,only:csr_from_dense,csr_matrix
  use sklearn_gp_kernels,only:rbf_covariance,white_covariance,operator(+)
  implicit none
  type(kernel_ridge)::model
  type(csr_matrix)::sx,sq
  real(dp),allocatable::x(:,:),query(:,:),y(:,:),truth(:,:),w(:),qw(:),penalty(:),pred(:,:),train(:,:)
  real(dp),allocatable::xcopy(:,:),qcopy(:,:),ycopy(:,:)
  integer::n,p,m,t,storage,kind,weight_mode,target_rank,alpha_mode,i,ios
  real(dp)::gamma,coef,weight_scalar,value
  character(16),parameter::names(12)=[character(16)::'linear','rbf','poly','sigmoid','cosine','laplacian', &
    'additive_chi2','chi2','precomputed','callable','gp','callable']
  do
    read(*,*,iostat=ios)n,p,m,t,storage,kind,weight_mode,target_rank,alpha_mode,gamma,coef
    if(ios/=0)exit
    allocate(x(n,p),query(m,p),y(n,t),truth(m,t),w(n),qw(m),penalty(t))
    do i=1,n
      read(*,*)x(i,:)
    end do
    do i=1,m
      read(*,*)query(i,:)
    end do
    do i=1,n
      read(*,*)y(i,:)
    end do
    do i=1,m
      read(*,*)truth(i,:)
    end do
    read(*,*)w;read(*,*)qw;read(*,*)penalty
    xcopy=x;qcopy=query;ycopy=y
    model%kernel=names(kind);model%gamma=gamma;model%coef0=coef;model%degree=3
    model%alpha=penalty(1)
    if(allocated(model%alpha_values))deallocate(model%alpha_values)
    if(alpha_mode/=0)model%alpha_values=penalty
    nullify(model%kernel_function)
    if(kind==10)model%kernel_function=>callback
    if(kind==12)model%kernel_function=>asymmetric_callback
    if(kind==11)model%kernel_object=rbf_covariance(1.2_dp)+white_covariance(.3_dp)
    weight_scalar=w(1)
    if(storage==1.or.storage==3)then
      sx=csr_from_dense(x)
      if(target_rank==0)then
        select case(weight_mode)
        case(0);call model%fit(sx,y(:,1))
        case(1);call model%fit(sx,y(:,1),w)
        case(2);call model%fit(sx,y(:,1),weight_scalar)
        end select
      else
        select case(weight_mode)
        case(0);call model%fit(sx,y)
        case(1);call model%fit(sx,y,w)
        case(2);call model%fit(sx,y,weight_scalar)
        end select
      end if
    else
      if(target_rank==0)then
        select case(weight_mode)
        case(0);call model%fit(x,y(:,1))
        case(1);call model%fit(x,y(:,1),w)
        case(2);call model%fit(x,y(:,1),weight_scalar)
        end select
      else
        select case(weight_mode)
        case(0);call model%fit(x,y)
        case(1);call model%fit(x,y,w)
        case(2);call model%fit(x,y,weight_scalar)
        end select
      end if
    end if
    if(storage==1.or.storage==2)then
      sq=csr_from_dense(query);pred=model%predict_multi(sq)
      if(target_rank==0)then
        value=model%score(sq,truth(:,1),qw)
        call require(maxval(abs(model%predict(sq)-pred(:,1)))==0,'vector/column kernel prediction')
      else
        value=model%score(sq,truth,qw)
      end if
    else
      pred=model%predict_multi(query)
      if(target_rank==0)then
        value=model%score(query,truth(:,1),qw)
        call require(maxval(abs(model%predict(query)-pred(:,1)))==0,'vector/column kernel prediction')
      else
        value=model%score(query,truth,qw)
      end if
    end if
    if(model%sparse_fit_)then
      train=model%predict_multi(model%x_fit_csr_)
    else
      train=model%predict_multi(model%x_fit_)
    end if
    call require(all(x==xcopy).and.all(query==qcopy).and.all(y==ycopy),'kernel ridge changed caller input')
    write(*,'(3(i0,1x),2(l1,1x))')model%n_samples_fit_,model%n_features_in_,model%n_targets_, &
      model%sparse_fit_,model%used_lstsq_
    write(*,'(*(es26.17e3,1x))')model%dual_coef_multi_
    write(*,'(*(es26.17e3,1x))')pred
    write(*,'(*(es26.17e3,1x))')train
    write(*,'(es26.17e3)')value
    deallocate(x,query,y,truth,w,qw,penalty)
  end do
contains
  real(dp)function callback(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    value=dot_product(a,b)+.5_dp
  end function
  real(dp)function asymmetric_callback(a,b)result(value)
    real(dp),intent(in)::a(:),b(:)
    value=dot_product(a,b)+.05_dp*sum(a)
  end function
end program
