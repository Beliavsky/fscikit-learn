module sklearn_tree
  use sklearn_legacy_rng, only: legacy_random_state
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean,ieee_argmax
  use, intrinsic :: iso_fortran_env, only: int64,real32
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_negative_inf
  use sklearn_common
  use sklearn_metrics, only: accuracy_score,r2_score
  implicit none
  private
  public :: decision_tree_classifier,decision_tree_regressor,extra_tree_classifier,extra_tree_regressor
  type :: tree_node
    integer :: left=0,right=0,feature=0,n_samples=0,depth=0
    logical :: missing_go_left=.false.
    real(dp) :: weighted_n_samples=0,threshold=0,impurity=0,value=0
    real(dp), allocatable :: probability(:)
  end type
  type :: tree_storage
    integer :: max_depth=100,min_samples_split=2,min_samples_leaf=1,max_features=0
    integer :: random_state=0,node_count_=0,n_features_in_=0
    logical :: random_thresholds=.false.
    logical, private :: weighted_fit_=.false.
    integer :: max_features_=0
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
  type, extends(decision_tree_classifier) :: extra_tree_classifier
  contains
    procedure :: fit=>extra_classifier_fit
  end type
  type, extends(decision_tree_regressor) :: extra_tree_regressor
  contains
    procedure :: fit=>extra_regressor_fit
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
    self%weighted_fit_=.false.; self%max_features_=self%max_features
    if(self%max_features_==0) self%max_features_=size(x,2)
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
      self%nodes_(node)%n_samples=n; self%nodes_(node)%weighted_n_samples=n; self%nodes_(node)%depth=depth
      parent=impurity(ids); self%nodes_(node)%impurity=parent
      if(size(classes)>0) then
        allocate(self%nodes_(node)%probability(size(classes)))
        do c=1,size(classes)
          self%nodes_(node)%probability(c)=real(count(nint(y(ids))==classes(c)),dp)/n
        end do
      else
        self%nodes_(node)%value=sum(y(ids))/n
      end if
      if(depth>=self%max_depth.or.n<self%min_samples_split.or.n/2<self%min_samples_leaf) return
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
          if(gain>best+epsilon(1._dp).or.(best_feature==0.and.gain>=0)) then
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
  ! Weighted/randomized splitter follows sklearn 1.6.1 tree/_splitter.pyx and _criterion.pyx.
  subroutine build_weighted_tree(self,x,y,classes,criterion,w,max_features)
    class(tree_storage), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:),w(:)
    integer, intent(in) :: classes(:)
    character(*), intent(in) :: criterion
    integer, intent(in), optional :: max_features
    type(legacy_random_state) :: rng
    real(dp), allocatable :: z(:,:)
    integer, allocatable :: active(:)
    integer :: all_features(size(x,2)),constants(size(x,2)),i,root,nf
    integer(int64) :: state
    real(dp) :: root_weight,min_weight,normalizer,contribution
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(size(w)==size(y),'tree sample weight length mismatch')
    call require(all(ieee_is_finite(w)),'tree sample weights must be finite')
    call require(size(y)<=huge(1)/2,'tree node capacity exceeds integer range')
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'tree training features exceed float32 range')
    call require(self%max_depth>=0.and.self%min_samples_split>=2.and.self%min_samples_leaf>=1, &
      'invalid tree size controls')
    call require(self%max_features>=0.and.self%max_features<=size(x,2),'invalid tree max_features')
    call require(ieee_is_finite(self%min_impurity_decrease),'tree impurity decrease must be finite')
    call require(self%min_impurity_decrease>=0,'negative impurity decrease')
    call require(self%random_state>=0,'weighted tree random_state must be nonnegative')
    nf=self%max_features; if(nf==0) nf=size(x,2)
    if(present(max_features)) nf=max_features
    self%max_features_=nf; self%weighted_fit_=.true.; self%n_features_in_=size(x,2); self%node_count_=0
    if(allocated(self%nodes_)) deallocate(self%nodes_)
    allocate(self%nodes_(2*size(y))); self%feature_importances_=spread(0._dp,1,size(x,2))
    z=real(real(x,real32),dp); active=pack([(i,i=1,size(y))],w/=0)
    root_weight=ieee_total(w); min_weight=ieee_product(0._dp,root_weight)
    all_features=[(i,i=1,size(x,2))]; constants=0
    call rng%seed(self%random_state); state=int(rng%integer_index(2147483647)-1,int64)
    root=grow(active,0,0,0._dp)
    do i=1,self%node_count_
      associate(node=>self%nodes_(i))
        if(node%left==0) cycle
        contribution=ieee_product(node%weighted_n_samples,node%impurity)
        contribution=ieee_difference(contribution, &
          ieee_product(self%nodes_(node%left)%weighted_n_samples,self%nodes_(node%left)%impurity))
        contribution=ieee_difference(contribution, &
          ieee_product(self%nodes_(node%right)%weighted_n_samples,self%nodes_(node%right)%impurity))
        self%feature_importances_(node%feature)=ieee_difference(self%feature_importances_(node%feature),-contribution)
      end associate
    end do
    do i=1,size(self%feature_importances_)
      self%feature_importances_(i)=ieee_quotient(self%feature_importances_(i),root_weight)
    end do
    normalizer=ieee_total(self%feature_importances_)
    if(.not.ieee_is_nan(normalizer)) then
      if(normalizer>0) self%feature_importances_=self%feature_importances_/normalizer
    end if
  contains
    integer function next_random() result(value)
      if(state==0) state=1
      state=iand(ieor(state,ishft(state,13)),4294967295_int64)
      state=ieor(state,ishft(state,-17))
      state=iand(ieor(state,ishft(state,5)),4294967295_int64)
      value=int(iand(state,2147483647_int64))
    end function
    subroutine swap(a,b)
      integer, intent(inout) :: a,b
      integer :: temporary
      temporary=a; a=b; b=temporary
    end subroutine
    logical function less(a,b) result(value)
      real(dp), intent(in) :: a,b
      value=.false.
      if(ieee_is_nan(a).or.ieee_is_nan(b)) return
      value=a<b
    end function
    subroutine moments(ids,mass,sums,squares)
      integer, intent(in) :: ids(:)
      real(dp), intent(out) :: mass,sums(:),squares
      real(dp) :: wy
      integer :: a,c,idx
      mass=0; sums=0; squares=0
      do a=1,size(ids)
        idx=ids(a); mass=ieee_difference(mass,-w(idx))
        if(size(classes)==0) then
          wy=ieee_product(w(idx),y(idx)); sums(1)=ieee_difference(sums(1),-wy)
          squares=ieee_difference(squares,-ieee_product(wy,y(idx)))
        else
          do c=1,size(classes)
            if(y(idx)==real(classes(c),dp)) sums(c)=ieee_difference(sums(c),-w(idx))
          end do
        end if
      end do
    end subroutine
    real(dp) function impurity(mass,sums,squares) result(value)
      real(dp), intent(in) :: mass,sums(:),squares
      real(dp) :: probability,term,total
      integer :: c
      if(size(classes)==0) then
        probability=ieee_quotient(sums(1),mass)
        value=ieee_difference(ieee_quotient(squares,mass),ieee_product(probability,probability))
      else if(criterion=='gini') then
        total=0
        do c=1,size(classes)
          total=ieee_difference(total,-ieee_product(sums(c),sums(c)))
        end do
        value=ieee_difference(1._dp,ieee_quotient(total,ieee_product(mass,mass)))
      else
        value=0
        do c=1,size(classes)
          if(ieee_is_nan(sums(c))) cycle
          if(sums(c)<=0) cycle
          probability=ieee_quotient(sums(c),mass)
          term=ieee_value(0._dp,ieee_quiet_nan)
          if(.not.ieee_is_nan(probability)) then
            if(probability>0) term=ieee_product(probability,log(probability)/log(2._dp))
          end if
          value=ieee_difference(value,term)
        end do
      end if
    end function
    recursive integer function grow(ids,depth,known,inherited) result(node)
      integer, intent(in) :: ids(:),depth,known
      real(dp), intent(in) :: inherited
      integer, allocatable :: ordered(:),left(:),right(:)
      real(dp) :: sums(max(1,size(classes))),sl(max(1,size(classes))),sr(max(1,size(classes)))
      real(dp) :: mass,squares,parent,wl,wr,sql,sqr,il,ir,proxy,best,low,high,threshold,best_threshold
      real(dp) :: best_left,best_right,best_wl,best_wr,improvement
      integer :: n,fi,fj,drawn,found,visited,nconstant,f,cut,j,c,best_feature,left_node,right_node,first,last
      n=size(ids); call moments(ids,mass,sums,squares)
      self%node_count_=self%node_count_+1; node=self%node_count_
      self%nodes_(node)%n_samples=n; self%nodes_(node)%weighted_n_samples=mass; self%nodes_(node)%depth=depth
      parent=inherited
      if(depth==0) parent=impurity(mass,sums,squares)
      self%nodes_(node)%impurity=parent
      if(size(classes)==0) then
        self%nodes_(node)%value=ieee_quotient(sums(1),mass)
      else
        allocate(self%nodes_(node)%probability(size(classes)))
        do c=1,size(classes)
          self%nodes_(node)%probability(c)=ieee_quotient(sums(c),mass)
        end do
      end if
      if(depth>=self%max_depth.or.n<self%min_samples_split.or.n/2<self%min_samples_leaf) return
      if(less(mass,ieee_product(2._dp,min_weight))) return
      if(.not.ieee_is_nan(parent)) then
        if(parent<=epsilon(1._dp)) return
      end if
      fi=size(x,2); found=0; drawn=0; visited=0; nconstant=known
      best=ieee_value(0._dp,ieee_negative_inf); best_feature=0; best_threshold=0
      best_left=0; best_right=0; best_wl=0; best_wr=0
      do while(fi>nconstant.and.(visited<nf.or.visited<=found+drawn))
        visited=visited+1; fj=drawn+modulo(next_random(),fi-found-drawn)
        if(fj<known) then
          call swap(all_features(drawn+1),all_features(fj+1)); drawn=drawn+1; cycle
        end if
        fj=fj+found; f=all_features(fj+1)
        ordered=ids(argsort(z(ids,f))); low=z(ordered(1),f); high=z(ordered(n),f)
        if(high<=real(real(low,real32)+1.e-7_real32,dp)) then
          call swap(all_features(fj+1),all_features(nconstant+1)); found=found+1; nconstant=nconstant+1; cycle
        end if
        fi=fi-1; call swap(all_features(fi+1),all_features(fj+1))
        first=self%min_samples_leaf; last=n-self%min_samples_leaf
        if(self%random_thresholds) then
          threshold=(high-low)*real(next_random(),dp)/2147483647._dp+low
          if(threshold==high) threshold=low
          cut=count(z(ordered,f)<=threshold); first=cut; last=cut
        end if
        do cut=first,last
          if(cut<self%min_samples_leaf.or.n-cut<self%min_samples_leaf) cycle
          if(.not.self%random_thresholds) then
            if(z(ordered(cut+1),f)<=real(real(z(ordered(cut),f),real32)+1.e-7_real32,dp)) cycle
            threshold=z(ordered(cut),f)/2+z(ordered(cut+1),f)/2
            if(threshold==z(ordered(cut+1),f)) threshold=z(ordered(cut),f)
          end if
          call moments(ordered(:cut),wl,sl,sql); wr=ieee_difference(mass,wl)
          if(less(wl,min_weight).or.less(wr,min_weight)) cycle
          do c=1,size(sums)
            sr(c)=ieee_difference(sums(c),sl(c))
          end do
          sqr=ieee_difference(squares,sql); il=impurity(wl,sl,sql); ir=impurity(wr,sr,sqr)
          if(size(classes)==0) then
            proxy=ieee_difference(ieee_quotient(ieee_product(sl(1),sl(1)),wl), &
              -ieee_quotient(ieee_product(sr(1),sr(1)),wr))
          else
            proxy=ieee_difference(-ieee_product(wr,ir),ieee_product(wl,il))
          end if
          if(ieee_is_nan(proxy)) cycle
          if(proxy>best) then
            best=proxy; best_feature=f; best_threshold=threshold
            best_left=il; best_right=ir; best_wl=wl; best_wr=wr
          end if
        end do
      end do
      all_features(:known)=constants(:known); constants(known+1:nconstant)=all_features(known+1:nconstant)
      if(best_feature==0) return
      improvement=ieee_difference(parent,ieee_product(ieee_quotient(best_wr,mass),best_right))
      improvement=ieee_difference(improvement,ieee_product(ieee_quotient(best_wl,mass),best_left))
      improvement=ieee_product(ieee_quotient(mass,root_weight),improvement)
      if(.not.ieee_is_nan(improvement)) then
        if(ieee_difference(improvement,-epsilon(1._dp))<self%min_impurity_decrease) return
      end if
      left=pack(ids,z(ids,best_feature)<=best_threshold); right=pack(ids,z(ids,best_feature)>best_threshold)
      self%nodes_(node)%feature=best_feature; self%nodes_(node)%threshold=best_threshold
      self%nodes_(node)%missing_go_left=size(left)>size(right)
      left_node=grow(left,depth+1,nconstant,best_left); right_node=grow(right,depth+1,nconstant,best_right)
      self%nodes_(node)%left=left_node; self%nodes_(node)%right=right_node
    end function
  end subroutine

  subroutine classifier_fit(self,x,y,sample_weight)
    class(decision_tree_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    call check_xy(x,size(y))
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'unknown tree criterion')
    self%classes_=unique_labels(y)
    if(present(sample_weight)) then
      call build_weighted_tree(self,x,real(y,dp),self%classes_,self%criterion,sample_weight)
    else
      call build_tree(self,x,real(y,dp),self%classes_,self%criterion)
    end if
  end subroutine
  subroutine regressor_fit(self,x,y,sample_weight)
    class(decision_tree_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    if(present(sample_weight)) then
      call build_weighted_tree(self,x,y,[integer::],'squared_error',sample_weight)
    else
      call build_tree(self,x,y,[integer::],'squared_error')
    end if
  end subroutine
  subroutine extra_classifier_fit(self,x,y,sample_weight)
    class(extra_tree_classifier), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(size(y))
    integer :: nf
    call check_xy(x,size(y)); self%random_thresholds=.true.
    call require(any(self%criterion==['gini    ','entropy ','log_loss']),'unknown tree criterion')
    self%classes_=unique_labels(y); weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'tree sample weight length mismatch'); weights=sample_weight
    end if
    nf=self%max_features; if(nf==0) nf=max(1,int(sqrt(real(size(x,2),dp))))
    call build_weighted_tree(self,x,real(y,dp),self%classes_,self%criterion,weights,nf)
  end subroutine
  subroutine extra_regressor_fit(self,x,y,sample_weight)
    class(extra_tree_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(size(y))
    call check_xy(x,size(y)); self%random_thresholds=.true.; weights=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==size(y),'tree sample weight length mismatch'); weights=sample_weight
    end if
    call build_weighted_tree(self,x,y,[integer::],'squared_error',weights)
  end subroutine
  function tree_apply(self,x) result(leaves)
    class(tree_storage), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: leaves(size(x,1)),i,node,f,j
    real(dp) :: value
    call require(allocated(self%nodes_),'decision tree is not fitted')
    if(self%weighted_fit_) then
      call require(size(x,1)>0.and.size(x,2)==self%n_features_in_,'tree query feature shape mismatch')
      call require(all(ieee_is_finite(x).or.ieee_is_nan(x)),'tree queries contain infinity')
      do j=1,size(x,2)
        do i=1,size(x,1)
          if(ieee_is_nan(x(i,j))) cycle
          call require(abs(x(i,j))<=real(huge(0._real32),dp),'tree query features exceed float32 range')
        end do
      end do
    else
      call check_xy(transpose(x),self%n_features_in_)
    end if
    do i=1,size(x,1)
      node=1
      do while(self%nodes_(node)%left>0)
        f=self%nodes_(node)%feature
        if(self%weighted_fit_) then
          if(ieee_is_nan(x(i,f))) then
            if(self%nodes_(node)%missing_go_left) then
              node=self%nodes_(node)%left
            else
              node=self%nodes_(node)%right
            end if
            cycle
          end if
          value=real(real(x(i,f),real32),dp)
        else
          value=x(i,f)
        end if
        if(value<=self%nodes_(node)%threshold) then
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
      if(self%weighted_fit_) then
        y(i)=self%classes_(ieee_argmax(prob(i,:)))
      else
        y(i)=self%classes_(maxloc(prob(i,:),dim=1))
      end if
    end do
  end function
  real(dp) function classifier_score(self,x,y,sample_weight) result(v)
    class(decision_tree_classifier), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer, intent(in) :: y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    integer, allocatable :: predicted(:)
    real(dp) :: mass
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'tree score target length mismatch')
    call check_score_weights(sample_weight,size(y))
    if(present(sample_weight)) then
      mass=ieee_total(sample_weight)
      if(.not.ieee_is_nan(mass)) call require(mass/=0,'tree score weights must have a nonzero sum')
      v=ieee_mean(merge(1._dp,0._dp,y==predicted),sample_weight)
    else
      v=accuracy_score(y,predicted)
    end if
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
  real(dp) function regressor_score(self,x,y,sample_weight) result(v)
    class(decision_tree_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: predicted(:),numerators(:),denominators(:)
    real(dp) :: center,num,den,mass
    integer :: i
    predicted=self%predict(x)
    call require(size(y)==size(predicted),'tree score target length mismatch')
    call require(all(ieee_is_finite(y)),'tree regression score targets must be finite')
    call require(all(ieee_is_finite(predicted)),'tree regression score predictions must be finite')
    call check_score_weights(sample_weight,size(y))
    if(size(y)<2) then
      v=ieee_value(0._dp,ieee_quiet_nan); return
    end if
    if(present(sample_weight)) then
      mass=ieee_total(sample_weight)
      if(.not.ieee_is_nan(mass)) call require(mass/=0,'tree score weights must have a nonzero sum')
    end if
    center=ieee_mean(y,sample_weight); allocate(numerators(size(y)),denominators(size(y)))
    do i=1,size(y)
      num=ieee_difference(y(i),predicted(i)); numerators(i)=ieee_product(num,num)
      den=ieee_difference(y(i),center); denominators(i)=ieee_product(den,den)
    end do
    num=ieee_mean(numerators,sample_weight); den=ieee_mean(denominators,sample_weight)
    v=ieee_value(0._dp,ieee_quiet_nan)
    if(ieee_is_nan(num).or.ieee_is_nan(den)) return
    v=1
    if(num/=0) then
      v=0
      if(den/=0) v=ieee_difference(1._dp,ieee_quotient(num,den))
    end if
  end function
  subroutine check_score_weights(weights,n)
    real(dp), intent(in), optional :: weights(:)
    integer, intent(in) :: n
    if(.not.present(weights)) return
    call require(size(weights)==n,'tree score weight length mismatch')
    call require(all(ieee_is_finite(weights)),'tree score weights must be finite')
  end subroutine
end module
