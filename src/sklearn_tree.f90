module sklearn_tree
  use sklearn_common
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: decision_tree_classifier,decision_tree_regressor
  type :: tree_node
    integer :: left=0,right=0,feature=0,n_samples=0,depth=0
    real(dp) :: threshold=0,impurity=0,value=0
    real(dp), allocatable :: probability(:)
  end type
  type :: tree_storage
    integer :: max_depth=100,min_samples_split=2,min_samples_leaf=1,max_features=0
    integer :: random_state=0,node_count_=0,n_features_in_=0
    logical :: random_thresholds=.false.
    real(dp) :: min_impurity_decrease=0
    real(dp), allocatable :: feature_importances_(:)
    type(tree_node), allocatable :: nodes_(:)
  contains
    procedure :: apply=>tree_apply
    procedure :: get_depth=>tree_depth
    procedure :: get_n_leaves=>tree_leaves
  end type
  type, extends(tree_storage) :: decision_tree_classifier
    character(16) :: criterion='gini'
    integer, allocatable :: classes_(:)
  contains
    procedure :: fit=>classifier_fit
    procedure :: predict_proba=>classifier_proba
    procedure :: predict=>classifier_predict
    procedure :: score=>classifier_score
  end type
  type, extends(tree_storage) :: decision_tree_regressor
  contains
    procedure :: fit=>regressor_fit
    procedure :: predict=>regressor_predict
    procedure :: score=>regressor_score
  end type
contains
  subroutine build_tree(self,x,y,classes,criterion)
    class(tree_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    integer, intent(in) :: classes(:)
    character(*), intent(in) :: criterion
    integer :: indices(size(y)),i,root
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%max_depth>=0.and.self%min_samples_split>=2.and.self%min_samples_leaf>=1, &
      'invalid tree size controls')
    call require(self%max_features>=0.and.self%max_features<=size(x,2),'invalid tree max_features')
    call require(self%min_impurity_decrease>=0,'negative impurity decrease')
    self%n_features_in_=size(x,2); self%node_count_=0
    if(allocated(self%nodes_)) deallocate(self%nodes_)
    allocate(self%nodes_(2*size(y))); self%feature_importances_=spread(0._dp,1,size(x,2))
    indices=[(i,i=1,size(y))]; call seed_rng(self%random_state)
    root=grow(indices,0)
    if(sum(self%feature_importances_)>0) &
      self%feature_importances_=self%feature_importances_/sum(self%feature_importances_)
  contains
    real(dp) function impurity(ids) result(v)
      integer, intent(in) :: ids(:)
      real(dp) :: probabilities(size(classes)),mean
      integer :: c
      if(size(classes)==0) then
        mean=sum(y(ids))/size(ids); v=sum((y(ids)-mean)**2)/size(ids)
      else
        do c=1,size(classes)
          probabilities(c)=real(count(nint(y(ids))==classes(c)),dp)/size(ids)
        end do
        if(criterion=='gini') then
          v=1-sum(probabilities**2)
        else
          v=0
          do c=1,size(classes)
            if(probabilities(c)>0) v=v-probabilities(c)*log(probabilities(c))/log(2._dp)
          end do
        end if
      end if
    end function
    recursive integer function grow(ids,depth) result(node)
      integer, intent(in) :: ids(:),depth
      integer, allocatable :: order(:),sorted(:),left(:),right(:),features(:)
      real(dp) :: parent,best,gain,threshold,best_threshold,r,random_cut
      integer :: n,p,j,k,feature,best_feature,best_cut,nf,a,b,tmp,c,left_node,right_node
      n=size(ids); p=size(x,2); self%node_count_=self%node_count_+1; node=self%node_count_
      self%nodes_(node)%n_samples=n; self%nodes_(node)%depth=depth
      parent=impurity(ids); self%nodes_(node)%impurity=parent
      if(size(classes)>0) then
        allocate(self%nodes_(node)%probability(size(classes)))
        do c=1,size(classes)
          self%nodes_(node)%probability(c)=real(count(nint(y(ids))==classes(c)),dp)/n
        end do
      else
        self%nodes_(node)%value=sum(y(ids))/n
      end if
      if(depth>=self%max_depth.or.n<self%min_samples_split.or.n<2*self%min_samples_leaf) return
      if(parent<=epsilon(1._dp)) return
      features=[(j,j=1,p)]; nf=p
      if(self%max_features>0) then
        nf=self%max_features
        do a=p,2,-1
          call random_number(r); b=1+int(r*a)
          tmp=features(a); features(a)=features(b); features(b)=tmp
        end do
      end if
      best=0; best_feature=0; best_threshold=0; best_cut=0
      do k=1,p
        feature=features(k); order=argsort(x(ids,feature)); sorted=ids(order)
        if(self%random_thresholds) then
          call random_number(r)
          random_cut=x(sorted(1),feature)+r*(x(sorted(n),feature)-x(sorted(1),feature))
        end if
        do j=self%min_samples_leaf,n-self%min_samples_leaf
          if(x(sorted(j),feature)==x(sorted(j+1),feature)) cycle
          threshold=x(sorted(j),feature)/2+x(sorted(j+1),feature)/2
          if(self%random_thresholds) then
            if(x(sorted(j),feature)>random_cut.or.x(sorted(j+1),feature)<=random_cut) cycle
            threshold=random_cut
          end if
          gain=parent-(j*impurity(sorted(:j))+(n-j)*impurity(sorted(j+1:)))/n
          if(gain>best+epsilon(1._dp)) then
            best=gain; best_feature=feature; best_threshold=threshold; best_cut=j
          end if
        end do
        ! Like sklearn, inspect additional features if sampled features cannot split.
        if(k>=nf.and.best_feature>0) exit
      end do
      if(best_feature==0) return
      if(best*n/size(y)<self%min_impurity_decrease) return
      left=pack(ids,x(ids,best_feature)<=best_threshold)
      right=pack(ids,x(ids,best_feature)>best_threshold)
      self%nodes_(node)%feature=best_feature; self%nodes_(node)%threshold=best_threshold
      self%feature_importances_(best_feature)=self%feature_importances_(best_feature)+n*best
      left_node=grow(left,depth+1); right_node=grow(right,depth+1)
      self%nodes_(node)%left=left_node; self%nodes_(node)%right=right_node
    end function
  end subroutine
  subroutine classifier_fit(self,x,y)
    class(decision_tree_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'unknown tree criterion')
    self%classes_=unique_labels(y)
    call build_tree(self,x,real(y,dp),self%classes_,self%criterion)
  end subroutine
  subroutine regressor_fit(self,x,y)
    class(decision_tree_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    call build_tree(self,x,y,[integer::],'squared_error')
  end subroutine
  function tree_apply(self,x) result(leaves)
    class(tree_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: leaves(size(x,1)),i,node,f
    call require(allocated(self%nodes_),'decision tree is not fitted')
    call check_xy(transpose(x),self%n_features_in_)
    do i=1,size(x,1)
      node=1
      do while(self%nodes_(node)%left>0)
        f=self%nodes_(node)%feature
        if(x(i,f)<=self%nodes_(node)%threshold) then
          node=self%nodes_(node)%left
        else
          node=self%nodes_(node)%right
        end if
      end do
      leaves(i)=node
    end do
  end function
  integer function tree_depth(self) result(depth)
    class(tree_storage), intent(in) :: self
    call require(allocated(self%nodes_),'decision tree is not fitted')
    depth=maxval(self%nodes_(:self%node_count_)%depth)
  end function
  integer function tree_leaves(self) result(n)
    class(tree_storage), intent(in) :: self
    call require(allocated(self%nodes_),'decision tree is not fitted')
    n=count(self%nodes_(:self%node_count_)%left==0)
  end function
  function classifier_proba(self,x) result(prob)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: prob(:,:)
    integer :: leaves(size(x,1)),i
    leaves=self%apply(x); allocate(prob(size(x,1),size(self%classes_)))
    do i=1,size(x,1)
      prob(i,:)=self%nodes_(leaves(i))%probability
    end do
  end function
  function classifier_predict(self,x) result(y)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: y(size(x,1)),i
    real(dp), allocatable :: prob(:,:)
    prob=self%predict_proba(x)
    do i=1,size(y)
      y(i)=self%classes_(maxloc(prob(i,:),dim=1))
    end do
  end function
  real(dp) function classifier_score(self,x,y) result(v)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    v=accuracy_score(y,self%predict(x))
  end function
  function regressor_predict(self,x) result(y)
    class(decision_tree_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: y(size(x,1))
    integer :: leaves(size(x,1)),i
    leaves=self%apply(x)
    do i=1,size(y)
      y(i)=self%nodes_(leaves(i))%value
    end do
  end function
  real(dp) function regressor_score(self,x,y) result(v)
    class(decision_tree_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
end module
