program reference_adaboost_cart_driver
  use sklearn_adaboost, only: adaboost_classifier
  use sklearn_tree, only: decision_tree_classifier
  use sklearn_ieee_math, only: ieee_argmax
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  type(adaboost_classifier) :: model
  real(dp), allocatable :: x(:,:),query(:,:),weights(:),score_weights(:),before(:,:),qbefore(:,:),wbefore(:)
  real(dp), allocatable :: cube(:,:,:),tree_importance(:,:),prediction(:,:),values(:)
  integer, allocatable :: y(:),labels(:),stages(:,:),seeds(:),depths(:),leaves(:)
  integer :: n,p,m,weighted,criterion,splitter,i,mode
  character(40) :: argument
  call get_command_argument(1,argument)
  if(len_trim(argument)>0) then
    call reject(trim(argument))
    if(trim(argument)=='features_large')then
      print *,'OVERSIZED_FEATURES_ACCEPTED';stop
    end if
    error stop 'invalid AdaBoost CART input accepted'
  end if
  read(*,*) mode,n,p,m,weighted,model%random_state,model%n_estimators,model%max_depth, &
    model%min_samples_split,model%min_samples_leaf,model%max_features,criterion,splitter, &
    model%learning_rate,model%min_impurity_decrease
  model%criterion='gini';if(criterion==1) model%criterion='entropy'
  if(criterion==2) model%criterion='log_loss'
  if(splitter/=0) model%splitter='random'
  allocate(x(n,p),query(m,p),y(n),weights(n),score_weights(n))
  read(*,*) x;read(*,*) query;read(*,*) y;read(*,*) weights;read(*,*) score_weights
  before=x;qbefore=query;wbefore=weights
  if(weighted/=0) then
    call model%fit(x,y,weights)
  else
    call model%fit(x,y)
  end if
  call integers(reshape(int([model%n_features_in_,model%n_classes_,model%n_estimators_,size(model%estimators_)],int64),[4,1]))
  call integers(reshape(int(model%classes_,int64),[model%n_classes_,1]))
  call matrix(reshape(model%estimator_weights_,[model%n_estimators,1]))
  call matrix(reshape(model%estimator_errors_,[model%n_estimators,1]))
  call matrix(reshape(model%feature_importances_,[p,1]))
  call matrix(model%decision_function(x));call matrix(model%decision_function(query))
  call matrix(model%predict_proba(query));call matrix(model%predict_log_proba(query))
  labels=model%predict(query);call integers(reshape(int(labels,int64),[m,1]))
  cube=model%staged_decision_function(query);call matrix(reshape(cube,[m*size(cube,2),size(cube,3)]))
  cube=model%staged_predict_proba(query);call matrix(reshape(cube,[m*size(cube,2),size(cube,3)]))
  stages=model%staged_predict(query);call integers(int(stages,int64))
  call matrix(reshape([model%score(x,y),model%score(x,y,score_weights)],[2,1]))
  values=model%staged_score(x,y);call matrix(reshape(values,[size(values),1]))
  values=model%staged_score(x,y,score_weights);call matrix(reshape(values,[size(values),1]))
  allocate(prediction(m,model%n_estimators_),tree_importance(p,model%n_estimators_))
  allocate(seeds(model%n_estimators_),depths(model%n_estimators_),leaves(model%n_estimators_))
  do i=1,model%n_estimators_
    labels=model%estimators_(i)%predict(query);prediction(:,i)=labels
    tree_importance(:,i)=model%estimators_(i)%feature_importances_
    seeds(i)=model%estimators_(i)%random_state;depths(i)=model%estimators_(i)%get_depth()
    leaves(i)=model%estimators_(i)%get_n_leaves()
  end do
  call integers(int(prediction,int64));call matrix(tree_importance)
  call integers(reshape(int(seeds,int64),[size(seeds),1]))
  call integers(reshape(int(depths,int64),[size(depths),1]))
  call integers(reshape(int(leaves,int64),[size(leaves),1]))
  call integers(reshape(int([model%estimator_%n_features_in_,model%estimator_%node_count_],int64),[2,1]))
  do i=1,model%n_estimators_
    call emit_tree(model%estimators_(i))
  end do
  if(mode==1) then
    call model%fit(x,y);call matrix(model%decision_function(query))
  end if
  call require(all(x==before).and.all(query==qbefore).and.all(weights==wbefore),'AdaBoost CART input changed')
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    print *, 'F',shape(a,kind=int64)
    print '(8(es25.16e3,1x))',a
  end subroutine
  subroutine integers(a)
    integer(int64), intent(in) :: a(:,:)
    print *, 'I',shape(a,kind=int64)
    print '(20(i0,1x))',a
  end subroutine
  subroutine emit_tree(tree)
    type(decision_tree_classifier), intent(in) :: tree
    integer(int64), allocatable :: nodes(:,:)
    real(dp), allocatable :: statistics(:,:),probability(:,:)
    integer :: n,i
    n=tree%node_count_;allocate(nodes(n,7),statistics(n,3),probability(n,size(tree%classes_)))
    nodes=0;statistics=0
    do i=1,n
      nodes(i,5)=tree%nodes_(i)%n_samples;nodes(i,6)=tree%nodes_(i)%depth
      statistics(i,2)=tree%nodes_(i)%weighted_n_samples;statistics(i,3)=tree%nodes_(i)%impurity
      probability(i,:)=tree%nodes_(i)%probability
      if(tree%nodes_(i)%left==0) then
        nodes(i,4)=tree%classes_(ieee_argmax(tree%nodes_(i)%probability))
      else
        nodes(i,1)=tree%nodes_(i)%feature;nodes(i,2)=tree%nodes_(i)%left;nodes(i,3)=tree%nodes_(i)%right
        nodes(i,7)=merge(1_int64,0_int64,tree%nodes_(i)%missing_go_left)
        statistics(i,1)=tree%nodes_(i)%threshold
      end if
    end do
    call integers(nodes);call matrix(statistics);call matrix(probability)
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    type(adaboost_classifier) :: bad
    real(dp) :: data(4,2),query(2,2)
    real(dp), allocatable :: output(:,:)
    integer :: labels(4)
    data(:,1)=[-2._dp,-1._dp,1._dp,2._dp];data(:,2)=0;query=0;labels=[-4,7,-4,7]
    select case(name)
    case('depth');bad%max_depth=0
    case('leaf');bad%min_samples_leaf=0
    case('split');bad%min_samples_split=1
    case('features_negative');bad%max_features=-1
    case('features_large');bad%max_features=3
    case('seed');bad%random_state=-1
    case('criterion');bad%criterion='bad'
    case('splitter');bad%splitter='bad'
    case('impurity_negative');bad%min_impurity_decrease=-1
    case('unfitted_log');output=bad%predict_log_proba(query);return
    case('unfitted_staged_score');block
      real(dp), allocatable :: score(:)
      score=bad%staged_score(query,[-4,7]);return
    end block
    case('score_weights_zero');block
      real(dp) :: score
      call bad%fit(data,labels);score=bad%score(data,labels,[0._dp,0._dp,0._dp,0._dp]);return
    end block
    case('score_weights_length');block
      real(dp) :: score
      call bad%fit(data,labels);score=bad%score(data,labels,[1._dp]);return
    end block
    case('score_target_length');block
      real(dp) :: score
      call bad%fit(data,labels);score=bad%score(data,[-4]);return
    end block
    case default;error stop 'unknown invalid AdaBoost CART case'
    end select
    call bad%fit(data,labels)
  end subroutine
end program
