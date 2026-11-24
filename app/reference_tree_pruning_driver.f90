program reference_tree_pruning_driver
  use sklearn_common, only: dp
  use sklearn_tree, only: decision_tree_classifier,extra_tree_classifier,decision_tree_regressor,extra_tree_regressor,tree_pruning_path
  use sklearn_ensemble, only: random_forest_classifier,extra_trees_classifier,random_forest_regressor,extra_trees_regressor
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  class(decision_tree_classifier), allocatable :: classifier
  class(decision_tree_regressor), allocatable :: regressor
  class(random_forest_classifier), allocatable :: classifier_forest
  class(random_forest_regressor), allocatable :: regressor_forest
  type(decision_tree_classifier), allocatable :: ct(:)
  type(decision_tree_regressor), allocatable :: rt(:)
  type(tree_pruning_path) :: path,before
  integer :: nc,c,kind,n,p,m,criterion,depth,leaf,nf,seed,nw,nt,boot,ns,cw,i,t,j,at,k,total,unchanged,nclass
  integer(int64) :: alpha_bits
  integer(int64), allocatable :: bits(:)
  integer, allocatable :: labels(:),predicted(:),classes(:),leaves(:,:),meta(:),left(:),right(:),feature(:),counts(:),tree_classes(:),seeds(:)
  real(dp), allocatable :: x(:,:),q(:,:),y(:),w(:),prediction(:),proba(:,:),importance(:),values(:),impurity(:),mass(:),threshold(:)
  character(16), parameter :: cc(3)=[character(16)::'gini','entropy','log_loss']
  character(16), parameter :: rc(4)=[character(16)::'squared_error','friedman_mse','poisson','absolute_error']
  character(20), parameter :: modes(3)=[character(20)::'none','balanced','balanced_subsample']
  read(*,*) nc
  do c=1,nc
    read(*,*) kind,n,p,m,criterion,depth,leaf,nf,seed,nw,alpha_bits,nt,boot,ns,cw
    allocate(x(n,p),q(m,p),y(n),bits(n*p));read(*,*) bits;x=reshape(transfer(bits,[0._dp],size(bits)),[n,p]);deallocate(bits)
    allocate(bits(n));read(*,*) bits;y=transfer(bits,y);deallocate(bits)
    allocate(bits(m*p));read(*,*) bits;q=reshape(transfer(bits,[0._dp],size(bits)),[m,p]);deallocate(bits)
    if(nw>=0) then
      allocate(w(nw),bits(nw));read(*,*) bits;w=transfer(bits,w);deallocate(bits)
    end if
    write(*,'(A,1X,I0)') 'BEGIN',c
    unchanged=1
    if(mod(kind,4)<2) then
      labels=int(y)
      if(kind<4) then
        if(kind==0) then
          allocate(decision_tree_classifier::classifier)
        else
          allocate(extra_tree_classifier::classifier)
        end if
        classifier%criterion=cc(criterion+1);classifier%max_depth=depth;classifier%min_samples_leaf=leaf
        classifier%max_features=nf;classifier%random_state=seed;classifier%ccp_alpha=transfer(alpha_bits,0._dp)
        before=classifier%cost_complexity_pruning_path(x,labels,w)
        if(allocated(classifier%nodes_)) unchanged=0
        call classifier%fit(x,labels,w);i=classifier%node_count_;prediction=real(classifier%predict(q),dp)
        path=classifier%cost_complexity_pruning_path(x,labels,w)
        if(classifier%node_count_/=i) unchanged=0
        if(any(classifier%predict(q)/=int(prediction))) unchanged=0
        if(any(path%ccp_alphas/=before%ccp_alphas)) unchanged=0
        predicted=classifier%predict(q);proba=classifier%predict_proba(q);importance=classifier%feature_importances_
        classes=classifier%classes_;allocate(ct(1));ct(1)=classifier
      else
        if(kind==4) then
          allocate(random_forest_classifier::classifier_forest)
        else
          allocate(extra_trees_classifier::classifier_forest)
        end if
        classifier_forest%criterion=cc(criterion+1);classifier_forest%max_depth=depth;classifier_forest%min_samples_leaf=leaf
        classifier_forest%max_features=nf;classifier_forest%random_state=seed;classifier_forest%ccp_alpha=transfer(alpha_bits,0._dp)
        classifier_forest%n_estimators=nt;classifier_forest%bootstrap=boot/=0;classifier_forest%max_samples=ns
        classifier_forest%class_weight=modes(cw+1);call classifier_forest%fit(x,labels,w)
        predicted=classifier_forest%predict(q);proba=classifier_forest%predict_proba(q);importance=classifier_forest%feature_importances_
        classes=classifier_forest%classes_;ct=classifier_forest%estimators_
        path%ccp_alphas=[real(dp)::];path%impurities=[real(dp)::]
      end if
      nclass=size(classes);nt=size(ct);allocate(leaves(m,nt),meta(5*nt),seeds(nt),tree_classes(nclass*nt))
      total=0;k=0
      do t=1,nt
        leaves(:,t)=ct(t)%apply(q);seeds(t)=ct(t)%random_state;total=total+ct(t)%node_count_
        meta(k+1:k+5)=[ct(t)%node_count_,ct(t)%get_depth(),ct(t)%get_n_leaves(),ct(t)%n_features_in_,ct(t)%max_features_];k=k+5
        tree_classes((t-1)*nclass+1:t*nclass)=ct(t)%classes_
      end do
      allocate(values(total*nclass),impurity(total),mass(total),threshold(total),left(total),right(total),feature(total),counts(total));at=0
      do t=1,nt
        do i=1,ct(t)%node_count_
          at=at+1
          associate(node=>ct(t)%nodes_(i))
            values((at-1)*nclass+1:at*nclass)=node%probability;impurity(at)=node%impurity;mass(at)=node%weighted_n_samples
            threshold(at)=node%threshold;left(at)=node%left;right(at)=node%right;feature(at)=node%feature;counts(at)=node%n_samples
          end associate
        end do
      end do
      call emit_i('prediction',predicted);call emit_r('proba',reshape(proba,[size(proba)]));call emit_i('classes',classes)
      call emit_i('tree_classes',tree_classes);deallocate(tree_classes)
    else
      if(kind<4) then
        if(kind==2) then
          allocate(decision_tree_regressor::regressor)
        else
          allocate(extra_tree_regressor::regressor)
        end if
        regressor%criterion=rc(criterion+1);regressor%max_depth=depth;regressor%min_samples_leaf=leaf
        regressor%max_features=nf;regressor%random_state=seed;regressor%ccp_alpha=transfer(alpha_bits,0._dp)
        before=regressor%cost_complexity_pruning_path(x,y,w)
        if(allocated(regressor%nodes_)) unchanged=0
        call regressor%fit(x,y,w);i=regressor%node_count_
        path=regressor%cost_complexity_pruning_path(x,y,w)
        if(regressor%node_count_/=i) unchanged=0
        prediction=regressor%predict(q);importance=regressor%feature_importances_;allocate(rt(1));rt(1)=regressor
      else
        if(kind==6) then
          allocate(random_forest_regressor::regressor_forest)
        else
          allocate(extra_trees_regressor::regressor_forest)
        end if
        regressor_forest%criterion=rc(criterion+1);regressor_forest%max_depth=depth;regressor_forest%min_samples_leaf=leaf
        regressor_forest%max_features=nf;regressor_forest%random_state=seed;regressor_forest%ccp_alpha=transfer(alpha_bits,0._dp)
        regressor_forest%n_estimators=nt;regressor_forest%bootstrap=boot/=0;regressor_forest%max_samples=ns
        call regressor_forest%fit(x,y,w);prediction=regressor_forest%predict(q);importance=regressor_forest%feature_importances_
        rt=regressor_forest%estimators_;path%ccp_alphas=[real(dp)::];path%impurities=[real(dp)::]
      end if
      nclass=0;nt=size(rt);allocate(leaves(m,nt),meta(5*nt),seeds(nt));total=0;k=0
      do t=1,nt
        leaves(:,t)=rt(t)%apply(q);seeds(t)=rt(t)%random_state;total=total+rt(t)%node_count_
        meta(k+1:k+5)=[rt(t)%node_count_,rt(t)%get_depth(),rt(t)%get_n_leaves(),rt(t)%n_features_in_,rt(t)%max_features_];k=k+5
      end do
      allocate(values(total),impurity(total),mass(total),threshold(total),left(total),right(total),feature(total),counts(total));at=0
      do t=1,nt
        do i=1,rt(t)%node_count_
          at=at+1
          associate(node=>rt(t)%nodes_(i))
            values(at)=node%value;impurity(at)=node%impurity;mass(at)=node%weighted_n_samples
            threshold(at)=node%threshold;left(at)=node%left;right(at)=node%right;feature(at)=node%feature;counts(at)=node%n_samples
          end associate
        end do
      end do
      call emit_r('prediction',prediction);call emit_r('proba',[real(dp)::]);call emit_i('classes',[integer::])
      call emit_i('tree_classes',[integer::])
    end if
    call emit_r('path_alphas',path%ccp_alphas);call emit_r('path_impurities',path%impurities)
    call emit_i('caller_preserved',[unchanged]);call emit_i('metadata',[p,nclass,nt])
    call emit_i('tree_metadata',meta);call emit_i('seeds',seeds);call emit_i('leaves',reshape(leaves,[size(leaves)]))
    call emit_r('importance',importance);call emit_r('values',values);call emit_r('impurity',impurity);call emit_r('mass',mass)
    call emit_r('threshold',threshold);call emit_i('left',left);call emit_i('right',right);call emit_i('feature',feature);call emit_i('count',counts)
    deallocate(x,q,y,values,impurity,mass,threshold,left,right,feature,counts,leaves,meta,seeds)
    if(allocated(w)) deallocate(w)
    if(allocated(classifier)) deallocate(classifier)
    if(allocated(regressor)) deallocate(regressor)
    if(allocated(classifier_forest)) deallocate(classifier_forest)
    if(allocated(regressor_forest)) deallocate(regressor_forest)
    if(allocated(ct)) deallocate(ct)
    if(allocated(rt)) deallocate(rt)
    write(*,'(A)') 'END'
  end do
contains
  subroutine emit_i(name,a)
    character(*), intent(in) :: name
    integer, intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,I0))') name,size(a),a
  end subroutine
  subroutine emit_r(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,ES26.17E3))') name,size(a),a
  end subroutine
end program
