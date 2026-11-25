program reference_bernoulli_options_driver
  use sklearn_common,only:dp
  use sklearn_naive_bayes,only:bernoulli_nb
  use sklearn_sparse,only:csr_matrix,csr_from_dense,csr_from_arrays
  implicit none
  type(bernoulli_nb)::model
  type(csr_matrix)::train,query
  real(dp),allocatable::x(:,:),q(:,:),w(:),prob(:,:),lp(:,:),joint(:,:)
  integer,allocatable::y(:),truth(:),classes(:),prediction(:)
  integer::n,p,m,k,storage,duplicates,stream,weighted,binarize,force,prior,ios,i,first,last,batches,batch
  real(dp)::alpha,threshold
  do
    read(*,*,iostat=ios)n,p,m,k,storage,duplicates,stream,weighted,binarize,force,prior,alpha,threshold
    if(ios/=0)exit
    allocate(x(n,p),q(m,p),w(n),y(n),truth(m),classes(k))
    do i=1,n
      read(*,*)x(i,:)
    end do
    do i=1,m
      read(*,*)q(i,:)
    end do
    read(*,*)y;read(*,*)truth;read(*,*)classes;read(*,*)w
    model%alpha=alpha;model%binarize=binarize==1;model%binarize_threshold=threshold
    model%force_alpha=force==1;model%fit_prior=prior/=1
    if(allocated(model%class_prior))deallocate(model%class_prior)
    if(prior==2)then
      model%class_prior=spread(1._dp/k,1,k)
      if(k>1)then
        model%class_prior(1)=0;model%class_prior(2:)=1._dp/(k-1)
      end if
    end if
    if(allocated(model%classes_))deallocate(model%classes_)
    if(allocated(model%feature_count_))deallocate(model%feature_count_)
    if(allocated(model%class_count_))deallocate(model%class_count_)
    if(allocated(model%feature_log_prob_))deallocate(model%feature_log_prob_)
    if(allocated(model%class_log_prior_))deallocate(model%class_log_prior_)
    model%n_features_in_=0
    batches=1;if(stream==1)batches=min(3,n)
    do batch=1,batches
      first=1+(batch-1)*n/batches;last=batch*n/batches
      if(storage==1.or.storage==3)train=sparse(x(first:last,:),duplicates==1)
      if(stream==0)then
        if(storage==1.or.storage==3)then
          if(weighted==1)then
            call model%fit(train,y,w)
          else
            call model%fit(train,y)
          end if
        else
          if(weighted==1)then
            call model%fit(x,y,w)
          else
            call model%fit(x,y)
          end if
        end if
      else
        if(storage==1.or.storage==3)then
          if(weighted==1)then
            call model%partial_fit(train,y(first:last),classes,w(first:last))
          else
            call model%partial_fit(train,y(first:last),classes)
          end if
        else
          if(weighted==1)then
            call model%partial_fit(x(first:last,:),y(first:last),classes,w(first:last))
          else
            call model%partial_fit(x(first:last,:),y(first:last),classes)
          end if
        end if
      end if
      if(storage==1.or.storage==2)then
        query=sparse(q,duplicates==1)
        joint=model%joint_log_likelihood(query);lp=model%predict_log_proba(query)
        prob=model%predict_proba(query);prediction=model%predict(query)
      else
        joint=model%joint_log_likelihood(q);lp=model%predict_log_proba(q)
        prob=model%predict_proba(q);prediction=model%predict(q)
      end if
      write(*,*)model%n_features_in_,size(model%classes_)
      write(*,*)model%classes_
      write(*,'(*(es26.17e3,1x))')model%class_count_
      write(*,'(*(es26.17e3,1x))')model%feature_count_
      write(*,'(*(es26.17e3,1x))')model%feature_log_prob_
      write(*,'(*(es26.17e3,1x))')model%class_log_prior_
      write(*,'(*(es26.17e3,1x))')joint
      write(*,'(*(es26.17e3,1x))')lp
      write(*,'(*(es26.17e3,1x))')prob
      write(*,*)prediction
    end do
    deallocate(x,q,w,y,truth,classes)
  end do
contains
  function sparse(a,duplicates)result(matrix)
    real(dp),intent(in)::a(:,:)
    logical,intent(in)::duplicates
    type(csr_matrix)::matrix
    integer,allocatable::ptr(:),col(:)
    real(dp),allocatable::value(:)
    integer::i,j,pos
    if(.not.duplicates)then
      matrix=csr_from_dense(a);return
    end if
    allocate(ptr(size(a,1)+1),col(2*size(a)),value(2*size(a)))
    pos=1;ptr(1)=1
    do i=1,size(a,1)
      do j=1,size(a,2)
        col(pos:pos+1)=j;value(pos:pos+1)=a(i,j)/2;pos=pos+2
      end do
      ptr(i+1)=pos
    end do
    matrix=csr_from_arrays(size(a,1),size(a,2),ptr,col,value)
  end function
end program
