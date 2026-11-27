program reference_huber_controls_driver
  use sklearn_linear_model,only:huber_regressor
  use sklearn_common,only:dp
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  implicit none
  integer::cases,c,snapshots,s,n,p,m,representation,query_rep,column,weights,intercept,warm
  integer::iterations
  real(dp)::epsilon,alpha,tol
  read(*,*)cases
  do c=1,cases
    block
      type(huber_regressor)::model
      type(csr_matrix)::sparse,query_sparse
      real(dp),allocatable::x(:,:),y(:),weight(:),query(:,:),prediction(:),train(:)
      real(dp)::scores(2)
      read(*,*)snapshots
      do s=1,snapshots
        read(*,*)n,p,m,representation,query_rep,column,weights,intercept,warm,epsilon,alpha,tol,iterations
        allocate(x(n,p),y(n),weight(n),query(m,p))
        read(*,*)x;read(*,*)y;read(*,*)weight;read(*,*)query
        sparse=matrix(x,representation);query_sparse=matrix(query,query_rep)
        model%fit_intercept=intercept/=0;model%warm_start=warm/=0
        model%epsilon=epsilon;model%alpha=alpha;model%tol=tol;model%max_iter=iterations
        if(representation==0)then
          if(column==0)then
            select case(weights)
            case(0);call model%fit(x,y)
            case(1);call model%fit(x,y,weight)
            case(2);call model%fit(x,y,weight(1))
            end select
          else
            select case(weights)
            case(0);call model%fit(x,reshape(y,[n,1]))
            case(1);call model%fit(x,reshape(y,[n,1]),weight)
            case(2);call model%fit(x,reshape(y,[n,1]),weight(1))
            end select
          end if
          train=model%predict(x)
          scores=[model%score(x,y),model%score(x,y,abs(weight)+.5_dp)]
        else
          if(column==0)then
            select case(weights)
            case(0);call model%fit(sparse,y)
            case(1);call model%fit(sparse,y,weight)
            case(2);call model%fit(sparse,y,weight(1))
            end select
          else
            select case(weights)
            case(0);call model%fit(sparse,reshape(y,[n,1]))
            case(1);call model%fit(sparse,reshape(y,[n,1]),weight)
            case(2);call model%fit(sparse,reshape(y,[n,1]),weight(1))
            end select
          end if
          train=model%predict(sparse)
          scores=[model%score(sparse,y),model%score(sparse,y,abs(weight)+.5_dp)]
        end if
        if(query_rep==0)then
          prediction=model%predict(query)
        else
          prediction=model%predict(query_sparse)
        end if
        print '(*(es25.16e3,1x))',model%coef_
        print '(*(es25.16e3,1x))',model%intercept_,model%scale_,model%loss_
        print '(*(es25.16e3,1x))',train
        print '(*(es25.16e3,1x))',prediction
        print '(*(i0,1x))',merge(1,0,model%outliers_)
        print '(*(i0,1x))',model%n_features_in_,model%n_iter_,merge(1,0,model%converged_)
        print '(*(es25.16e3,1x))',scores
        deallocate(x,y,weight,query)
      end do
    end block
  end do
contains
  function matrix(x,style)result(csr)
    real(dp),intent(in)::x(:,:)
    integer,intent(in)::style
    type(csr_matrix)::csr
    integer,allocatable::ptr(:),indices(:)
    real(dp),allocatable::data(:)
    integer::i,j,k,n,p
    if(style<2)then
      csr=csr_from_dense(x);return
    end if
    n=size(x,1);p=size(x,2);allocate(ptr(n+1),indices(2*n*p+n),data(2*n*p+n));k=0;ptr(1)=1
    do i=1,n
      do j=p,1,-1
        if(x(i,j)==0)cycle
        k=k+1;indices(k)=j;data(k)=.25_dp*x(i,j)
        k=k+1;indices(k)=j;data(k)=.75_dp*x(i,j)
      end do
      k=k+1;indices(k)=1;data(k)=0;ptr(i+1)=k+1
    end do
    csr=csr_from_arrays(n,p,ptr,indices(:k),data(:k))
  end function
end program
