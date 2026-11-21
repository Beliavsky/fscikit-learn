! IsolationForest: sklearn 1.6.1 ensemble/_iforest.py, _bagging.py and dense random tree contracts.
! BSD notices retained in LICENSES and THIRD_PARTY_NOTICES.md; native Fortran implementation.
module sklearn_isolation_forest
  use sklearn_bagging_sampling, only: sample_indices
  use sklearn_common, only: dp,require,check_x,argsort
  use sklearn_tree, only: decision_tree_regressor
  use sklearn_legacy_rng, only: legacy_random_state
  use iso_fortran_env, only: int64,real32
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: isolation_forest
  type :: sampled_indices
    integer, allocatable :: values(:)
  end type
  type :: isolation_forest
    integer :: n_estimators=100,max_samples=0,max_features=0,random_state=0
    integer :: n_estimators_=0,n_features_in_=0,max_samples_=0,max_features_=0
    real(dp) :: max_samples_fraction=0,max_features_fraction=0,contamination=-1,offset_=-.5_dp
    logical :: bootstrap=.false.,warm_start=.false.
    type(decision_tree_regressor), allocatable :: estimators_(:)
    type(sampled_indices), allocatable :: estimators_features_(:),estimators_samples_(:)
  contains
    procedure :: fit
    procedure :: score_samples
    procedure :: decision_function
    procedure :: predict
    procedure :: fit_predict
  end type
contains
  function float_features(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call check_x(x)
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'IsolationForest features exceed float32 range')
    z=real(real(x,real32),dp)
  end function
  subroutine fit(self,x,sample_weight)
    class(isolation_forest), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(legacy_random_state) :: target_rng,seed_rng,bag_rng,tree_rng
    type(decision_tree_regressor), allocatable :: trees(:)
    type(sampled_indices), allocatable :: features(:),samples(:)
    real(dp) :: z(size(x,1),size(x,2)),y(size(x,1)),w(size(x,1)),tree_weights(size(x,1))
    real(dp), allocatable :: scores(:)
    integer, allocatable :: order(:)
    real(dp) :: position
    integer :: n,p,m,nf,old,i,k,seed,tree_seed,depth,lo,hi
    z=float_features(x); n=size(x,1); p=size(x,2)
    call require(self%n_estimators>0.and.self%random_state>=0,'invalid IsolationForest estimator count or seed')
    call require(all(ieee_is_finite([self%max_samples_fraction,self%max_features_fraction,self%contamination])), &
      'nonfinite IsolationForest controls')
    call require(self%max_samples>=0.and.self%max_features>=0,'negative IsolationForest sampling count')
    call require(self%max_samples_fraction>=0.and.self%max_samples_fraction<=1.and. &
      self%max_features_fraction>=0.and.self%max_features_fraction<=1,'invalid IsolationForest sampling fraction')
    call require(.not.(self%max_samples>0.and.self%max_samples_fraction>0).and. &
      .not.(self%max_features>0.and.self%max_features_fraction>0),'IsolationForest sampling count/fraction conflict')
    call require(self%contamination==-1.or.(self%contamination>0.and.self%contamination<=.5_dp), &
      'IsolationForest contamination must be -1 (auto) or in (0,.5]')
    m=min(256,n)
    if(self%max_samples>0) m=min(self%max_samples,n)
    if(self%max_samples_fraction>0) m=int(self%max_samples_fraction*n)
    call require(m>0.or.self%bootstrap,'IsolationForest sampling fraction selects no rows without bootstrap')
    nf=p
    if(self%max_features>0) nf=self%max_features
    if(self%max_features_fraction>0) nf=max(1,int(self%max_features_fraction*p))
    call require(nf<=p,'IsolationForest max_features exceeds input features')
    w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'IsolationForest sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)),'IsolationForest weights must be finite and nonnegative')
      call require(all(sample_weight>=0),'IsolationForest weights must be finite and nonnegative')
      w=sample_weight
    end if
    old=0
    if(self%warm_start.and.allocated(self%estimators_)) then
      old=size(self%estimators_)
      call require(p==self%n_features_in_,'IsolationForest warm-start feature count mismatch')
      call require(self%n_estimators>=old,'IsolationForest warm_start cannot reduce n_estimators')
    end if
    allocate(trees(self%n_estimators),features(self%n_estimators),samples(self%n_estimators))
    if(old>0) then
      trees(:old)=self%estimators_; features(:old)=self%estimators_features_; samples(:old)=self%estimators_samples_
    end if
    call target_rng%seed(self%random_state)
    do i=1,n
      y(i)=target_rng%uniform()
    end do
    call seed_rng%seed(self%random_state)
    do k=1,old
      seed=seed_rng%integer_index(2147483647)-1
    end do
    depth=0
    do while(2._dp**depth<max(m,2))
      depth=depth+1
    end do
    do k=old+1,self%n_estimators
      seed=seed_rng%integer_index(2147483647)-1
      call tree_rng%seed(seed); tree_seed=tree_rng%integer_index(2147483647)-1
      call bag_rng%seed(seed)
      features(k)%values=sample_indices(bag_rng,p,nf,.false.)
      samples(k)%values=sample_indices(bag_rng,n,m,self%bootstrap)
      tree_weights=0
      if(self%bootstrap) then
        do i=1,m
          tree_weights(samples(k)%values(i))=tree_weights(samples(k)%values(i))+1
        end do
        tree_weights=tree_weights*w
      else
        tree_weights(samples(k)%values)=w(samples(k)%values)
      end if
      call require(all(ieee_is_finite(tree_weights)),'IsolationForest bootstrap weight overflow')
      trees(k)%max_depth=depth; trees(k)%max_features=1; trees(k)%random_thresholds=.true.
      trees(k)%random_state=tree_seed
      call random_tree(trees(k),z(:,features(k)%values),y,tree_weights)
    end do
    call move_alloc(trees,self%estimators_); call move_alloc(features,self%estimators_features_)
    call move_alloc(samples,self%estimators_samples_)
    self%n_estimators_=self%n_estimators; self%n_features_in_=p; self%max_samples_=m; self%max_features_=nf
    self%offset_=-.5_dp
    if(self%contamination/=-1) then
      scores=self%score_samples(z); order=argsort(scores); position=1+(n-1)*self%contamination
      lo=floor(position); hi=ceiling(position)
      self%offset_=scores(order(lo))+(position-lo)*(scores(order(hi))-scores(order(lo)))
    end if
  end subroutine
  real(dp) function path_length(n) result(value)
    integer, intent(in) :: n
    value=0
    if(n==2) then
      value=1
    else if(n>2) then
      value=2*(log(real(n-1,dp))+.577215664901532860606512090082_dp)-2*real(n-1,dp)/n
    end if
  end function
  function score_samples(self,x) result(scores)
    class(isolation_forest), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1)),z(size(x,1),size(x,2)),denominator
    integer :: leaves(size(x,1)),k,i,node
    call require(allocated(self%estimators_),'IsolationForest is not fitted')
    call require(size(x,2)==self%n_features_in_,'IsolationForest query feature count mismatch')
    z=float_features(x); scores=0
    do k=1,size(self%estimators_)
      leaves=self%estimators_(k)%apply(z(:,self%estimators_features_(k)%values))
      do i=1,size(scores)
        node=leaves(i)
        scores(i)=scores(i)+(real(self%estimators_(k)%nodes_(node)%depth+1,dp)+ &
          path_length(self%estimators_(k)%nodes_(node)%n_samples)-1)
      end do
    end do
    denominator=size(self%estimators_)*path_length(self%max_samples_)
    if(denominator==0) then
      scores=-.5_dp
    else
      scores=-2._dp**(-scores/denominator)
    end if
  end function
  function decision_function(self,x) result(scores)
    class(isolation_forest), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: scores(size(x,1))
    scores=self%score_samples(x)-self%offset_
  end function
  function predict(self,x) result(labels)
    class(isolation_forest), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    integer :: labels(size(x,1))
    labels=merge(1,-1,self%decision_function(x)>=0)
  end function
  function fit_predict(self,x,sample_weight) result(labels)
    class(isolation_forest), intent(inout) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), optional, intent(in) :: sample_weight(:)
    integer :: labels(size(x,1))
    call self%fit(x,sample_weight); labels=self%predict(x)
  end function
  subroutine random_tree(tree,x,y,w)
    type(decision_tree_regressor), intent(inout) :: tree
    real(dp), intent(in) :: x(:,:),y(:),w(:)
    type(legacy_random_state) :: rng
    integer, allocatable :: samples(:)
    integer :: features(size(x,2)),constants(size(x,2)),i,root
    integer(int64) :: state
    real(dp) :: root_weight
    samples=pack([(i,i=1,size(y))],w>0); features=[(i,i=1,size(x,2))]; constants=0
    call rng%seed(tree%random_state); state=int(rng%integer_index(2147483647)-1,int64)
    allocate(tree%nodes_(max(1,2*size(samples)))); tree%node_count_=0; tree%n_features_in_=size(x,2)
    tree%feature_importances_=spread(0._dp,1,size(x,2)); root_weight=sum(w)
    root=grow(1,size(samples),0,0,0._dp)
    tree%nodes_=tree%nodes_(:tree%node_count_)
    if(sum(tree%feature_importances_)>0) &
      tree%feature_importances_=tree%feature_importances_/sum(tree%feature_importances_)
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
      integer :: temp
      temp=a; a=b; b=temp
    end subroutine
    recursive integer function grow(first,last,depth,known,inherited) result(node)
      integer, intent(in) :: first,last,depth,known
      real(dp), intent(in) :: inherited
      real(dp) :: mass,total,squares,parent,low,high,threshold,wl,sl,sql,wr,il,ir,gain,value
      integer :: a,b,n,f,fi,fj,drawn,found,visited,nconstant,cut,leftnode,rightnode,idx
      n=last-first+1; mass=0; total=0; squares=0
      do a=first,last
        idx=samples(a); mass=mass+w(idx); total=total+w(idx)*y(idx); squares=squares+w(idx)*y(idx)**2
      end do
      tree%node_count_=tree%node_count_+1; node=tree%node_count_
      tree%nodes_(node)%n_samples=n; tree%nodes_(node)%weighted_n_samples=mass; tree%nodes_(node)%depth=depth
      if(n==0) return ! Finite native convention for sklearn's empty weighted root.
      parent=inherited
      if(depth==0) parent=squares/mass-(total/mass)**2
      tree%nodes_(node)%value=total/mass; tree%nodes_(node)%impurity=parent
      if(n<2.or.depth>=tree%max_depth.or.parent<=epsilon(1._dp)) return
      fi=size(x,2); found=0; drawn=0; visited=0; nconstant=known; f=0; threshold=0
      do while(fi>nconstant.and.(visited<1.or.visited<=found+drawn))
        visited=visited+1; fj=drawn+modulo(next_random(),fi-found-drawn)
        if(fj<known) then
          call swap(features(drawn+1),features(fj+1)); drawn=drawn+1
          cycle
        end if
        fj=fj+found; f=features(fj+1)
        low=minval(x(samples(first:last),f)); high=maxval(x(samples(first:last),f))
        if(high<=real(real(low,real32)+1.e-7_real32,dp)) then
          call swap(features(fj+1),features(nconstant+1)); found=found+1; nconstant=nconstant+1; f=0
          cycle
        end if
        fi=fi-1; call swap(features(fi+1),features(fj+1))
        threshold=(high-low)*real(next_random(),dp)/2147483647._dp+low
        if(threshold==high) threshold=low
        a=first; b=last
        do while(a<=b)
          if(x(samples(a),f)<=threshold) then
            a=a+1
          else
            call swap(samples(a),samples(b)); b=b-1
          end if
        end do
        cut=a-1
      end do
      features(:known)=constants(:known); constants(known+1:nconstant)=features(known+1:nconstant)
      if(f==0) return
      if(cut<first.or.cut>=last) return
      wl=0; sl=0; sql=0
      do a=first,cut
        idx=samples(a); wl=wl+w(idx); sql=sql+w(idx)*y(idx)**2
      end do
      if(cut-first+1<=last-cut) then
        do a=first,cut
          idx=samples(a); sl=sl+w(idx)*y(idx)
        end do
      else
        sl=total; wl=mass
        do a=last,cut+1,-1
          idx=samples(a); sl=sl-w(idx)*y(idx); wl=wl-w(idx)
        end do
      end if
      wr=mass-wl; il=sql/wl-(sl/wl)**2; ir=(squares-sql)/wr-((total-sl)/wr)**2
      value=parent-wr/mass*ir-wl/mass*il; gain=mass*parent-wl*il-wr*ir
      if(mass/root_weight*value+epsilon(1._dp)<0) return
      tree%nodes_(node)%feature=f; tree%nodes_(node)%threshold=threshold
      tree%feature_importances_(f)=tree%feature_importances_(f)+gain
      leftnode=grow(first,cut,depth+1,nconstant,il); rightnode=grow(cut+1,last,depth+1,nconstant,ir)
      tree%nodes_(node)%left=leftnode; tree%nodes_(node)%right=rightnode
    end function
  end subroutine
end module
