program reference_pipeline_driver
  use reference_pipeline_models
  implicit none
  type(pipeline) :: model,prefix
  type(feature_string), allocatable :: input(:),names(:)
  real(dp), allocatable :: x(:,:),q(:,:),yr(:),ym(:,:),weights(:),fit(:,:),train(:,:),query(:,:),prediction(:,:), &
    training_prediction(:,:),proba(:,:),decision(:,:),inverse(:,:),scores(:),vector(:)
  integer, allocatable :: yi(:),predicted(:),neighbor_index(:,:)
  real(dp), allocatable :: neighbor_distance(:,:)
  integer :: task,kind,config,n,p,m,named,weighted,real_class,i,k
  character(32) :: buffer
  read(*,*) task,kind,config,n,p,m,named,weighted,real_class
  allocate(x(n,p),q(m,p),yr(n),ym(n,2),yi(n),weights(n),input(p))
  read(*,*) x;read(*,*) q;read(*,*) yr;read(*,*) yi;read(*,*) ym;read(*,*) weights
  do i=1,p
    write(buffer,'(a,i0)') 'field_',i;input(i)%text=trim(buffer)
  end do
  call configure(task,kind,config,model);forward_calls=0;inverse_calls=0
  if(task==0) then
    if(mod(config,2)==1) then
      if(named==1) then
        fit=model%fit_transform(x,yr,input)
      else
        fit=model%fit_transform(x,yr)
      end if
    else
      if(named==1) then
        call model%fit(x,yr,feature_names=input)
      else
        call model%fit(x,yr)
      end if
      fit=model%transform(x)
    end if
    prefix=model
  else
    call fit_model()
    prefix=model%slice(1,size(model%steps)-1)
    fit=prefix%transform(x)
  end if
  train=prefix%transform(x);query=prefix%transform(q)
  allocate(proba(0,0),decision(0,0))
  if(task==0) then
    prediction=model%transform(q);training_prediction=model%transform(x);allocate(scores(0))
  else if(task==2) then
    predicted=model%predict_class(q);prediction=reshape(real(predicted,dp),[size(predicted),1])
    predicted=model%predict_class(x);training_prediction=reshape(real(predicted,dp),[size(predicted),1])
    proba=model%predict_proba(q)
    if(config==1) then
      deallocate(decision);decision=model%decision_function(q)
    end if
    allocate(scores(2));scores(1)=model%score(x,yi);scores(2)=model%score(x,yi,weights)
  else
    prediction=model%predict_matrix(q);training_prediction=model%predict_matrix(x)
    if(task==4) then
      allocate(scores(1));scores(1)=model%score(x,yr)
    else
      allocate(scores(2))
      if(task==3) then
        scores(1)=model%score(x,ym);scores(2)=model%score(x,ym,weights)
      else
        scores(1)=model%score(x,yr);scores(2)=model%score(x,yr,weights)
      end if
    end if
  end if
  if(kind>=13) then
    names=prefix%get_feature_names_out(input)
  else
    names=prefix%get_feature_names_out()
  end if
  ! The reference fixture suite only requests inverse calls for these kinds.
  ! SimpleImputer has the method, but requires add_indicator=True to call it.
  if(prefix%can_inverse().and.any(kind==[0,2,3,4,8,9,10,11,13,14])) then
    inverse=prefix%inverse_transform(train)
  else
    allocate(inverse(0,0))
  end if
  k=0;if(allocated(model%classes_)) k=size(model%classes_)
  print *,model%n_features_in_,model%estimator_kind_,model%target_rank_,forward_calls,inverse_calls,k
  if(k>0) then
    print *,model%classes_
  else
    print *
  end if
  call emit(fit);call emit(train);call emit(query);call emit(prediction);call emit(training_prediction)
  call emit(proba);call emit(decision);call emit(inverse)
  print *,size(scores);print '(*(es25.16e3,1x))',scores
  print *,size(names)
  do i=1,size(names)
    print '(a)',names(i)%text
  end do
  if(config==2.and.(task==1.or.task==2)) then
    print *,1
    select type(tail=>model%steps(size(model%steps))%model)
    type is(kneighbors_regressor)
      call emit(tail%search_%x_);call emit(reshape(tail%y_,[n,1]))
      call tail%search_%kneighbors(query,neighbor_distance,neighbor_index)
      call emit(neighbor_distance);call emit(real(neighbor_index,dp))
      call tail%search_%kneighbors(train,neighbor_distance,neighbor_index)
      call emit(neighbor_distance);call emit(real(neighbor_index,dp))
    type is(kneighbors_classifier)
      call emit(tail%search_%x_);call emit(reshape(real(tail%y_,dp),[n,1]))
      call tail%search_%kneighbors(query,neighbor_distance,neighbor_index)
      call emit(neighbor_distance);call emit(real(neighbor_index,dp))
      call tail%search_%kneighbors(train,neighbor_distance,neighbor_index)
      call emit(neighbor_distance);call emit(real(neighbor_index,dp))
    class default
      error stop 'expected KNN final model'
    end select
  else
    print *,0
  end if
  if(config==3.and.task>0) then
    select type(tail=>model%steps(size(model%steps))%model)
    type is(pipeline)
      if(allocated(tail%feature_names_in_)) then
        print *,size(tail%feature_names_in_)
        do i=1,size(tail%feature_names_in_)
          print '(a)',tail%feature_names_in_(i)%text
        end do
      else
        print *,0
      end if
    class default
      error stop 'expected nested final pipeline'
    end select
  end if
  if(kind==14) then
    select type(tail=>model%steps(size(model%steps))%model)
    type is(pipeline)
      print *,merge(1,0,tail%steps(1)%integer_target_)
      select type(selection=>tail%steps(1)%model)
      type is(select_k_best)
        print '(*(es25.16e3,1x))',selection%scores_
      end select
    end select
  end if
contains
  subroutine fit_model()
    type(feature_string), allocatable :: supplied(:)
    real(dp), allocatable :: w(:)
    if(named==1) supplied=input
    if(weighted==1) w=weights
    select case(task)
    case(1,4)
      if(kind==14) then
        call model%fit(x,yi,w,supplied)
      else
        call model%fit(x,yr,w,supplied)
      end if
    case(2)
      if(real_class==1) then
        call model%fit(x,real(yi,dp),w,supplied)
      else
        call model%fit(x,yi,w,supplied)
      end if
    case(3);call model%fit(x,ym,w,supplied)
    end select
  end subroutine
  subroutine emit(matrix)
    real(dp), intent(in) :: matrix(:,:)
    print *,shape(matrix);print '(*(es25.16e3,1x))',matrix
  end subroutine
end program
