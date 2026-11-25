program reference_dummy_extended_driver
  use sklearn_common,only:dp,require
  use sklearn_dummy
  use sklearn_sparse,only:csr_matrix,csr_from_dense
  use,intrinsic::ieee_arithmetic
  implicit none
  type(dummy_classifier)::classifier
  type(dummy_regressor)::regressor
  type(dummy_probability),allocatable::proba(:)
  type(csr_matrix)::xs,qs
  real(dp),allocatable::x(:,:),query(:,:),y(:,:),truth(:,:),weights(:),qw(:),constants(:),pred(:,:)
  integer,allocatable::labels(:,:),actual(:,:),class_constants(:)
  integer::n,p,m,qp,t,family,storage,rank,weight_mode,strategy,seed,score_mode,ios,i,j
  real(dp)::q,score
  character(16),parameter::class_strategies(5)=[character(16)::'prior','most_frequent','stratified','uniform','constant']
  character(16),parameter::reg_strategies(4)=[character(16)::'mean','median','quantile','constant']
  do
    read(*,*,iostat=ios)n,p,m,qp,t,family,storage,rank,weight_mode,strategy,seed,score_mode,q
    if(ios/=0)exit
    allocate(x(n,p),query(m,qp),y(n,t),truth(m,t),weights(n),qw(m),constants(t),labels(n,t),class_constants(t))
    x=ieee_value(0._dp,ieee_quiet_nan)
    query=ieee_value(0._dp,ieee_quiet_nan)
    do i=1,n
      read(*,*)y(i,:)
    end do
    do i=1,m
      read(*,*)truth(i,:)
    end do
    read(*,*)weights;read(*,*)qw;read(*,*)constants
    if(storage==1)then
      xs=csr_from_dense(x);qs=csr_from_dense(query)
    end if
    if(family==0)then
      labels=int(y);class_constants=int(constants)
      classifier%strategy=class_strategies(strategy);classifier%random_state=seed
      if(allocated(classifier%constant_values))deallocate(classifier%constant_values)
      if(strategy==5)classifier%constant_values=class_constants
      if(storage==0)then
        if(rank==0)then
          select case(weight_mode)
          case(0);call classifier%fit(x,labels(:,1))
          case(1);call classifier%fit(x,labels(:,1),weights)
          case(2);call classifier%fit(x,labels(:,1),weights(1))
          end select
        else
          select case(weight_mode)
          case(0);call classifier%fit(x,labels)
          case(1);call classifier%fit(x,labels,weights)
          case(2);call classifier%fit(x,labels,weights(1))
          end select
        end if
        proba=classifier%predict_proba_multi(query);actual=classifier%predict_multi(query)
        if(t==1)then
          call require(all(classifier%predict(query)==actual(:,1)),'dummy vector prediction adapter')

        end if
      else
        if(rank==0)then
          select case(weight_mode)
          case(0);call classifier%fit(xs,labels(:,1))
          case(1);call classifier%fit(xs,labels(:,1),weights)
          case(2);call classifier%fit(xs,labels(:,1),weights(1))
          end select
        else
          select case(weight_mode)
          case(0);call classifier%fit(xs,labels)
          case(1);call classifier%fit(xs,labels,weights)
          case(2);call classifier%fit(xs,labels,weights(1))
          end select
        end if
        proba=classifier%predict_proba_multi(qs);actual=classifier%predict_multi(qs)
        if(t==1)call require(all(classifier%predict(qs)==actual(:,1)),'dummy sparse vector adapter')
      end if
      write(*,*)classifier%n_outputs_,classifier%n_features_in_,classifier%n_classes_multi_
      do j=1,t
        write(*,*)classifier%distributions_(j)%classes
        write(*,'(*(es26.17e3,1x))')classifier%distributions_(j)%prior
        write(*,'(*(es26.17e3,1x))')proba(j)%values
      end do
      write(*,*)actual
      score=0
      if(score_mode==1)then
        if(storage==0)then
          if(rank==0)then
            score=classifier%score(query,int(truth(:,1)),qw)
          else
            score=classifier%score(query,int(truth),qw)
          end if
        else
          if(rank==0)then
            score=classifier%score(qs,int(truth(:,1)),qw)
          else
            score=classifier%score(qs,int(truth),qw)
          end if
        end if
      end if
    else
      regressor%strategy=reg_strategies(strategy);regressor%quantile=q
      if(allocated(regressor%constant_values))deallocate(regressor%constant_values)
      if(strategy==4)regressor%constant_values=constants
      if(storage==0)then
        if(rank==0)then
          select case(weight_mode)
          case(0);call regressor%fit(x,y(:,1))
          case(1);call regressor%fit(x,y(:,1),weights)
          case(2);call regressor%fit(x,y(:,1),weights(1))
          end select
          score=regressor%score(query,truth(:,1),qw)
        else
          select case(weight_mode)
          case(0);call regressor%fit(x,y)
          case(1);call regressor%fit(x,y,weights)
          case(2);call regressor%fit(x,y,weights(1))
          end select
          score=regressor%score(query,truth,qw)
        end if
        pred=regressor%predict_multi(query)
        if(t==1)call require(all(regressor%predict(query)==pred(:,1)),'dummy regression vector adapter')
      else
        if(rank==0)then
          select case(weight_mode)
          case(0);call regressor%fit(xs,y(:,1))
          case(1);call regressor%fit(xs,y(:,1),weights)
          case(2);call regressor%fit(xs,y(:,1),weights(1))
          end select
          score=regressor%score(qs,truth(:,1),qw)
        else
          select case(weight_mode)
          case(0);call regressor%fit(xs,y)
          case(1);call regressor%fit(xs,y,weights)
          case(2);call regressor%fit(xs,y,weights(1))
          end select
          score=regressor%score(qs,truth,qw)
        end if
        pred=regressor%predict_multi(qs)
        if(t==1)call require(all(regressor%predict(qs)==pred(:,1)),'dummy sparse regression vector adapter')
      end if
      write(*,*)regressor%n_outputs_,regressor%n_features_in_
      write(*,'(*(es26.17e3,1x))')regressor%constants_
      write(*,'(*(es26.17e3,1x))')pred
    end if
    write(*,'(es26.17e3)')score
    deallocate(x,query,y,truth,weights,qw,constants,labels,class_constants)
  end do
end program
